from __future__ import annotations

import csv
import gzip
import importlib
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


RISK_REQUIRED_INPUTS = ("mri_image", "synthseg_labels", "tumor_mask")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK_PROJECT_ROOT = Path(
    os.environ.get(
        "CAPSTONE_MRI_PROJECT_ROOT",
        PROJECT_ROOT / "5_surgical_risk_mapping",
    )
).expanduser().resolve()
DEFAULT_RISK_CONFIG = RISK_PROJECT_ROOT / "configs" / "config_v7.json"


def risk_prerequisite_status(
    config_path: Path | None, model_mask: np.ndarray | None
) -> dict[str, Any]:
    """The legacy V8 mapper needs anatomical labels; BraTS region masks are not substitutes."""
    reasons = []
    if model_mask is None:
        reasons.append("3D tumor mask is unavailable.")
    if config_path is None:
        reasons.append(
            "V8 risk mapping requires an explicit config referencing a registered MRI, "
            "SynthSeg anatomical labels, and tumor mask."
        )
    elif not config_path.is_file():
        reasons.append(f"Risk configuration does not exist: {config_path.name}")
    else:
        import json

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            reasons.append(
                f"Risk configuration could not be read ({type(error).__name__})."
            )
        else:
            if not isinstance(config, dict):
                reasons.append("Risk configuration must be a JSON object.")
            else:
                paths = config.get("paths", {})
                if not isinstance(paths, dict):
                    reasons.append("Risk config field 'paths' must be a JSON object.")
                else:
                    for key in RISK_REQUIRED_INPUTS:
                        configured_path = paths.get(key)
                        if not isinstance(configured_path, str) or not configured_path.strip():
                            reasons.append(f"Risk config is missing required paths.{key}.")
                            continue
                        path = Path(configured_path)
                        resolved_path = (
                            path if path.is_absolute() else RISK_PROJECT_ROOT / path
                        )
                        if not resolved_path.is_file():
                            reasons.append(
                                f"Risk prerequisite paths.{key} does not exist: {resolved_path.name}"
                            )
    return {
        "status": "blocked" if reasons else "not run",
        "preflight": "incomplete" if reasons else "configured; risk calculation not executed",
        "reasons": reasons,
        "note": (
            "The mapper's SynthSeg structure labels and required spatial QC cannot be "
            "replaced with the BraTS WT/TC/ET/RC segmentation ontology. This app does not "
            "execute the risk mapper or bind the configured tumor-mask path to its prediction."
        ),
    }


def _reorient_to_ras(source: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, np.ndarray]:
    orientation = nib.orientations.io_orientation(source.affine)
    target = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orientation, target)
    array = nib.orientations.apply_orientation(source.get_fdata(), transform)
    affine = source.affine @ nib.orientations.inv_ornt_aff(transform, source.shape)
    return np.asarray(array), np.asarray(affine)


def _reorient_from_ras(array: np.ndarray, affine: np.ndarray) -> np.ndarray:
    source_orientation = nib.orientations.io_orientation(affine)
    ras_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(ras_orientation, source_orientation)
    return np.asarray(nib.orientations.apply_orientation(array, transform))


def _nifti_gzip_bytes(
    array: np.ndarray, affine: np.ndarray, header: nib.nifti1.Nifti1Header, dtype: np.dtype
) -> bytes:
    image = nib.Nifti1Image(np.asarray(array, dtype=dtype), affine, header.copy())
    image.header.set_data_dtype(dtype)
    return gzip.compress(image.to_bytes())


def run_v8_risk_mapping(
    mri_path: Path,
    tumor_mask_path: Path,
    synthseg_path: Path,
    output_dir: Path,
    patient_id: str,
    config_template: Path | None = None,
) -> dict[str, Any]:
    """Run the existing V8 engine with explicit, registered inputs and isolated outputs."""
    template_path = config_template or DEFAULT_RISK_CONFIG
    if not template_path.is_file():
        raise FileNotFoundError(f"V8 configuration template not found: {template_path.name}")

    mri = nib.load(str(mri_path))
    labels = nib.load(str(synthseg_path))
    tumor = nib.load(str(tumor_mask_path))
    if len(mri.shape) != 3 or len(labels.shape) != 3 or len(tumor.shape) != 3:
        raise ValueError("V8 MRI, SynthSeg, and tumor-mask inputs must all be 3D.")
    if labels.shape != mri.shape or not np.allclose(labels.affine, mri.affine, atol=1e-4):
        raise ValueError(
            "Uploaded SynthSeg labels must match the uploaded t1c MRI shape and affine."
        )

    mri_ras, affine_ras = _reorient_to_ras(mri)
    labels_ras, labels_affine = _reorient_to_ras(labels)
    tumor_ras, tumor_affine = _reorient_to_ras(tumor)
    if labels_ras.shape != mri_ras.shape or not np.allclose(
        labels_affine, affine_ras, atol=1e-4
    ):
        raise ValueError("MRI and SynthSeg labels did not preserve alignment after RAS reorientation.")
    if tumor_ras.shape != mri_ras.shape or not np.allclose(
        tumor_affine, affine_ras, atol=1e-4
    ):
        raise ValueError(
            "3D tumor mask and reoriented MRI do not share shape and affine; refusing risk mapping."
        )
    if (
        not np.isfinite(mri_ras).all()
        or not np.isfinite(labels_ras).all()
        or not np.isfinite(tumor_ras).all()
    ):
        raise ValueError("V8 MRI, SynthSeg, and tumour-mask volumes must contain only finite voxel values.")
    if not np.any(labels_ras > 0):
        raise ValueError("SynthSeg labels contain no non-background anatomy.")
    label_values = np.unique(labels_ras)
    if not np.allclose(label_values, np.round(label_values), atol=1e-4):
        raise ValueError("SynthSeg label volume contains non-integer values.")
    pipeline_path = str((RISK_PROJECT_ROOT / "pipeline").resolve())
    sys.path.insert(0, pipeline_path)
    try:
        from label_mapping import SYNTHSEG_LABELS
    finally:
        sys.path.remove(pipeline_path)

    valid_labels = set(SYNTHSEG_LABELS) | set(range(1000, 3000))
    unknown_labels = sorted(set(np.rint(label_values).astype(int)) - valid_labels)
    if unknown_labels:
        raise ValueError(
            "Uploaded label volume contains values outside the V8 SynthSeg/FreeSurfer "
            f"label mapping: {unknown_labels[:12]}"
        )
    if not np.any(tumor_ras > 0):
        raise ValueError("3D segmentation produced an empty tumor mask; V8 cannot run.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v8_inputs_", dir=output_dir) as working:
        work = Path(working)
        risk_mri = work / "mri_t1c_ras.nii.gz"
        risk_labels = work / "synthseg_ras.nii.gz"
        risk_tumor = work / "tumor_mask_ras.nii.gz"
        nib.save(nib.Nifti1Image(mri_ras.astype(np.float32), affine_ras), risk_mri)
        nib.save(
            nib.Nifti1Image(np.rint(labels_ras).astype(np.int32), affine_ras),
            risk_labels,
        )
        nib.save(
            nib.Nifti1Image((tumor_ras > 0).astype(np.uint8), affine_ras),
            risk_tumor,
        )

        config = json.loads(template_path.read_text(encoding="utf-8"))
        config["patient_id"] = patient_id
        paths = config.setdefault("paths", {})
        paths.update(
            {
                "mri_image": str(risk_mri.resolve()),
                "tumor_mask": str(risk_tumor.resolve()),
                "synthseg_labels": str(risk_labels.resolve()),
                "dti_fa": None,
                "tract_masks": None,
                "output_dir_v6": str((output_dir / "v8_outputs" / "V6_baseline").resolve()),
                "output_dir_v7_validated": str(
                    (output_dir / "v8_outputs" / "V7_VALIDATED").resolve()
                ),
                "output_dir_v8_corridors": str(
                    (output_dir / "v8_outputs" / "V8_CORRIDORS").resolve()
                ),
                "output_dir_v8_ablation": str(
                    (output_dir / "v8_outputs" / "V8_ablation").resolve()
                ),
                "output_dir_v8_comparison": str(
                    (output_dir / "v8_outputs" / "V8_comparison").resolve()
                ),
                "output_dir_v8_qc": str(
                    (output_dir / "v8_outputs" / "V8_qc").resolve()
                ),
            }
        )
        config_path = work / "v8_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        pipeline_path = str((RISK_PROJECT_ROOT / "pipeline").resolve())
        root_path = str(RISK_PROJECT_ROOT.resolve())
        sys.path.insert(0, pipeline_path)
        sys.path.insert(0, root_path)
        try:
            v8 = importlib.import_module("run_v8_pipeline")
            with redirect_stdout(StringIO()):
                summary = v8.run_pipeline(str(config_path))
        finally:
            sys.path.remove(root_path)
            sys.path.remove(pipeline_path)

        validated_dir = Path(paths["output_dir_v7_validated"])
        corridors_dir = Path(paths["output_dir_v8_corridors"])
        risk_image = nib.load(str(validated_dir / "enhanced_risk_map.nii.gz"))
        risk_map = np.asarray(risk_image.get_fdata(dtype=np.float32))
        expected_ras_shape = tumor_ras.shape
        if (
            risk_map.shape != expected_ras_shape
            or not np.allclose(risk_image.affine, tumor_affine, atol=1e-4)
        ):
            raise RuntimeError(
                "V8 risk-map output does not match the reoriented input geometry."
            )

        corridor_file = corridors_dir / "corridor_results.csv"
        if corridor_file.is_file():
            with corridor_file.open(newline="", encoding="utf-8") as stream:
                corridors = list(csv.DictReader(stream))
        else:
            corridors = []
        corridor_mask_path = corridors_dir / "candidate_corridors.nii.gz"
        corridor_mask = (
            np.asarray(nib.load(str(corridor_mask_path)).get_fdata())
            if corridor_mask_path.is_file()
            else None
        )
        if corridor_mask is not None and corridor_mask.shape != expected_ras_shape:
            raise RuntimeError("V8 corridor mask does not match the input volume shape.")
        component_maps = {}
        for component in ("G", "S", "T", "C"):
            component_path = validated_dir / f"{component}_map.nii.gz"
            if component_path.is_file():
                component_image = nib.load(str(component_path))
                component_data = np.asarray(
                    component_image.get_fdata(dtype=np.float32)
                )
                if (
                    component_data.shape != expected_ras_shape
                    or not np.allclose(component_image.affine, tumor_affine, atol=1e-4)
                ):
                    raise RuntimeError(
                        f"V8 {component} component map does not match input geometry."
                    )
                component_maps[component] = component_data
        risk_map_original = _reorient_from_ras(risk_map, tumor.affine)
        if corridor_mask is not None:
            corridor_mask = _reorient_from_ras(corridor_mask, tumor.affine)
        component_maps = {
            name: _reorient_from_ras(component, tumor.affine)
            for name, component in component_maps.items()
        }
        brain_mask = _reorient_from_ras((labels_ras > 0) | (tumor_ras > 0), tumor.affine)
        interactive_report_path = corridors_dir / "risk_3d_viewer.html"
        risk_nifti_bytes = _nifti_gzip_bytes(
            risk_map_original, tumor.affine, tumor.header, np.dtype(np.float32)
        )
        corridor_nifti_bytes = (
            _nifti_gzip_bytes(
                corridor_mask.astype(np.uint8),
                tumor.affine,
                tumor.header,
                np.dtype(np.uint8),
            )
            if corridor_mask is not None
            else None
        )
        return {
            "status": "completed",
            "risk_map": risk_map_original,
            "component_maps": component_maps,
            "brain_mask": brain_mask,
            "corridor_mask": corridor_mask,
            "corridors": corridors,
            "structures": summary["structure_records"],
            "evidence_status": summary["evidence_status"],
            "active_weights": summary["active_weights"],
            "qc": summary["qc"],
            "output_affine": np.asarray(tumor.affine).copy(),
            "output_paths": {
                "validated": str(validated_dir),
                "corridors": str(corridors_dir),
            },
            "risk_map_nifti": risk_nifti_bytes,
            "corridor_mask_nifti": corridor_nifti_bytes,
            "interactive_report_html": (
                interactive_report_path.read_bytes()
                if interactive_report_path.is_file()
                else None
            ),
            "note": (
                "V8 research-prototype composite score; not a clinical probability. "
                "DTI/tract inputs were not supplied; those components are unavailable "
                "and omitted from the active weights."
            ),
        }


def write_uploaded_synthseg(upload: object, destination: Path) -> Path:
    """Persist a supplied SynthSeg NIfTI in the isolated run directory."""
    import gzip

    if isinstance(upload, (bytes, bytearray, memoryview)):
        payload = upload
    elif callable(get_buffer := getattr(upload, "getbuffer", None)):
        payload = get_buffer()
    else:
        raise TypeError("SynthSeg upload must be bytes or provide a getbuffer() method.")
    if not payload:
        raise ValueError("SynthSeg label upload is empty.")
    with destination.open("wb") as stream:
        if payload[:2] == b"\x1f\x8b":
            stream.write(payload)
        else:
            with gzip.GzipFile(fileobj=stream, mode="wb") as compressed:
                compressed.write(payload)
    return destination
