from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import nibabel as nib
import numpy as np

from integrated_pipeline.modules.mri_io import write_uploaded_study
from integrated_pipeline.modules.model_adapters import classify_slice, segment_slice
from integrated_pipeline.modules.risk_adapter import (
    run_v8_risk_mapping,
    write_uploaded_synthseg,
)
from integrated_pipeline.modules.slice_extraction import extract_axial_slice, select_slice


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(
    case_id: str,
    uploaded: Mapping[str, object],
    checkpoint_3d: Path,
    checkpoint_2d_segmentation: Path | None = None,
    checkpoint_classification: Path | None = None,
    slice_index: int | None = None,
    risk_config: Path | None = None,
    synthseg_upload: object | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run available local model stages; keep unsupported risk inputs explicit."""
    def update_progress(message: str) -> None:
        if progress is not None:
            progress(message)

    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", case_id.strip())
    if not safe_id or safe_id != case_id.strip():
        raise ValueError("Study identifier may contain only letters, digits, '_' and '-'.")
    with tempfile.TemporaryDirectory(prefix="mri_integrated_") as temporary:
        root = Path(temporary)
        input_dir = root / safe_id
        update_progress("Validating the four MRI volumes and spatial metadata.")
        study = write_uploaded_study(safe_id, uploaded, input_dir)
        case_dir = input_dir
        output_dir = root / "segmentation3d"
        checkpoint_3d = Path(checkpoint_3d).expanduser().resolve()

        from src.segmentation3d.monai_engine import predict_case
        from src.data.brats import BraTSCase, load_input_volume

        update_progress("Running the selected 3D segmentation checkpoint.")
        predict_case(checkpoint_3d, case_dir, output_dir)
        prediction_metadata = json.loads(
            (output_dir / "prediction_metadata.json").read_text(encoding="utf-8")
        )
        orientation = prediction_metadata.get("orientation")
        if orientation != "RAS":
            raise ValueError(
                f"Integrated slice extraction currently requires RAS model orientation; "
                f"checkpoint uses {orientation!r}."
            )
        model_modalities = tuple(prediction_metadata["modalities"])
        case = BraTSCase(
            case_id=safe_id,
            patient_group=safe_id.rsplit("-", 1)[0],
            directory=case_dir,
        )
        update_progress("Extracting the tumor-centre axial slice and model outputs.")
        volume = load_input_volume(
            case, modalities=model_modalities, orientation=orientation
        )
        predicted_masks: dict[str, np.ndarray] = {}
        for region in ("WT", "TC", "ET", "RC"):
            mask_path = output_dir / f"{safe_id}_{region}.nii.gz"
            if mask_path.is_file():
                predicted_masks[region] = np.asarray(nib.load(str(mask_path)).get_fdata() > 0)
        if "WT" not in predicted_masks:
            raise RuntimeError("3D predictor completed without a WT mask.")
        if tuple(volume["image"].shape[1:]) != predicted_masks["WT"].shape:
            raise RuntimeError(
                "3D source volume and saved prediction mask have different spatial shapes."
            )
        output_image = nib.load(str(output_dir / f"{safe_id}_WT.nii.gz"))
        if not np.allclose(volume["affine"], output_image.affine, atol=1e-4):
            raise RuntimeError(
                "3D source volume and saved prediction mask have different voxel-to-world affines."
            )

        selected_slice = (
            select_slice(predicted_masks["WT"], tuple(predicted_masks["WT"].shape))
            if slice_index is None
            else int(slice_index)
        )
        modality_index = model_modalities.index("t1c")
        source_slice = extract_axial_slice(volume["image"][modality_index].numpy(), selected_slice)
        output_affine = np.asarray(output_image.affine)
        output_spacing = tuple(float(value) for value in nib.affines.voxel_sizes(output_affine))
        resolved_config = prediction_metadata.get("resolved_config", {})
        report_config = {
            key: resolved_config[key]
            for key in (
                "model",
                "modalities",
                "region_order",
                "label_scheme",
                "orientation",
                "patch_size",
                "device",
            )
            if key in resolved_config
        }
        risk_result = None
        risk_status = {
            "status": "blocked",
            "reasons": [
                "Upload a SynthSeg label volume registered to the uploaded t1c MRI "
                "to run the legacy V8 risk mapper."
            ],
            "note": (
                "BraTS WT/TC/ET/RC masks are not anatomical SynthSeg labels. "
                "Risk mapping is not inferred from those masks."
            ),
        }
        if synthseg_upload is not None:
            update_progress(
                "Running legacy V8 risk mapping with strict registration QC; "
                "DTI/tract metrics will remain unavailable."
            )
            try:
                synthseg_path = write_uploaded_synthseg(
                    synthseg_upload, root / f"{safe_id}-synthseg.nii.gz"
                )
                risk_result = run_v8_risk_mapping(
                    case_dir / f"{safe_id}-t1c.nii.gz",
                    output_dir / f"{safe_id}_WT.nii.gz",
                    synthseg_path,
                    root / "risk_mapping",
                    safe_id,
                    risk_config,
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                KeyError,
                IndexError,
                TypeError,
                ImportError,
            ) as error:
                risk_status = {
                    "status": "failed",
                    "reasons": [str(error)],
                    "note": "V8 did not produce a result that passed input or spatial checks.",
                }
            else:
                risk_status = {
                    "status": "completed",
                    "qc": risk_result["qc"],
                    "active_weights": risk_result["active_weights"],
                    "evidence_status": risk_result["evidence_status"],
                    "note": risk_result["note"],
                }
        result: dict[str, Any] = {
            "case_id": safe_id,
            "stages": {
                "3D segmentation": {"status": "completed"},
                "surgical risk mapping": risk_status,
                "2D slice extraction": {"status": "completed"},
                "2D segmentation": {
                    "status": "not configured" if checkpoint_2d_segmentation is None else "pending"
                },
                "2D classification": {
                    "status": "not configured" if checkpoint_classification is None else "pending"
                },
                "RAG": {
                    "status": "not run",
                    "note": (
                        "An authorized query can send structured outputs from this run "
                        "as request-scoped context; raw MRI is not sent."
                    ),
                },
            },
            "model_3d": prediction_metadata["pipeline"],
            "model_configuration": report_config,
            "inference_device": prediction_metadata.get("device", "not recorded"),
            "software_versions": {
                "torch": prediction_metadata.get("torch_version", "not recorded"),
                "monai": prediction_metadata.get("monai_version", "not recorded"),
            },
            "seed": prediction_metadata.get("seed", "not recorded"),
            "prediction_threshold": prediction_metadata.get("prediction_threshold", "not recorded"),
            "region_definitions": prediction_metadata.get("region_definitions", {}),
            "input_shape": list(study.shape),
            "output_shape": list(predicted_masks["WT"].shape),
            "input_spacing_mm": list(study.spacing),
            "output_spacing_mm": list(output_spacing),
            "input_affine": study.affine.tolist(),
            "output_affine": output_affine.tolist(),
            "orientation": orientation,
            "modalities": list(model_modalities),
            "region_masks": predicted_masks,
            "risk_mapping": risk_result,
            "slice_image": source_slice,
            "slice_index": selected_slice,
            "classification": None,
            "segmentation2d": None,
            "segmentation2d_metadata": None,
        }

        if checkpoint_2d_segmentation is not None:
            update_progress("Running the optional 2D segmentation model.")
            try:
                result["segmentation2d"], result["segmentation2d_metadata"] = segment_slice(
                    Path(checkpoint_2d_segmentation).expanduser().resolve(), source_slice
                )
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                result["stages"]["2D segmentation"] = {
                    "status": "failed",
                    "model": Path(checkpoint_2d_segmentation).name,
                    "reasons": [str(error)],
                }
            else:
                result["stages"]["2D segmentation"] = {
                    "status": "completed",
                    "model": Path(checkpoint_2d_segmentation).name,
                    "reasons": [
                        "BRISC 2D checkpoint applied to an MRI slice; domain compatibility is unvalidated."
                    ],
                }
                result["model_2d_segmentation"] = Path(checkpoint_2d_segmentation).name
        if checkpoint_classification is not None:
            update_progress("Running the optional 2D classification model.")
            try:
                result["classification"] = classify_slice(
                    Path(checkpoint_classification).expanduser().resolve(), source_slice
                )
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                result["stages"]["2D classification"] = {
                    "status": "failed",
                    "model": Path(checkpoint_classification).name,
                    "reasons": [str(error)],
                }
            else:
                result["stages"]["2D classification"] = {
                    "status": "completed",
                    "model": Path(checkpoint_classification).name,
                    "reasons": [
                        "BRISC slice classifier output is not a diagnosis; MRI-to-BRISC domain compatibility is unvalidated."
                    ],
                }
                result["model_2d_classification"] = Path(checkpoint_classification).name
        update_progress("Pipeline stages finished; review blocked and failed stages.")
        return result
