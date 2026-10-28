"""Build a shareable, self-contained case report that embeds the 3D case viewer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.clinical_case_report import generate_clinical_case_report

AXIS_NAMES = {"L": "left_right", "R": "left_right", "A": "front_back", "P": "front_back",
              "S": "up_down", "I": "up_down"}


def describe_tumor(mri_path: Path, mask_path: Path) -> Dict[str, Any]:
    """Estimate tumor side and overall size from image geometry."""
    mri_image = nib.load(str(mri_path))
    mask_image = nib.load(str(mask_path))
    mask = np.asarray(mask_image.dataobj) > 0
    if mask.shape != mri_image.shape[:3] or not mask.any():
        raise ValueError("The tumor mask must be non-empty and match the MRI grid.")
    brain = np.asarray(mri_image.dataobj) != 0
    codes = nib.aff2axcodes(mri_image.affine)
    spacing = mask_image.header.get_zooms()[:3]

    extent: Dict[str, float] = {}
    side: Optional[str] = None
    crosses_midline = False
    for axis, code in enumerate(codes):
        other = tuple(a for a in range(3) if a != axis)
        indices = np.flatnonzero(mask.any(axis=other))
        extent[AXIS_NAMES[code]] = float((indices.max() - indices.min() + 1) * spacing[axis])
        if code in ("L", "R"):
            brain_indices = np.flatnonzero(brain.any(axis=other))
            midline = (brain_indices.min() + brain_indices.max()) / 2
            centroid = float(np.argwhere(mask)[:, axis].mean())
            toward_increasing = "left" if code == "L" else "right"
            toward_decreasing = "right" if code == "L" else "left"
            side = toward_increasing if centroid > midline else toward_decreasing
            crosses_midline = bool(indices.min() < midline < indices.max())
    return {"side": side, "crosses_midline": crosses_midline, "extent_mm": extent}


def build_report(
    config_path: Path,
    summary_path: Path,
    corridors_path: Path,
    viewer_path: Path,
    output_dir: Path,
    case_id: str,
    age: Optional[int],
    sex: Optional[str],
    history: str,
    sample_note: Optional[str],
    classification: Optional[Dict[str, Any]] = None,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    corridors = json.loads(corridors_path.read_text(encoding="utf-8")).get("corridors")
    viewer_html = viewer_path.read_text(encoding="utf-8")
    tumor_details = describe_tumor(
        PROJECT_ROOT / paths["mri_image"], PROJECT_ROOT / paths["tumor_mask"]
    )
    patient_details: Dict[str, Any] = {"history": history}
    if age is not None:
        patient_details["age"] = age
    if sex:
        patient_details["sex"] = sex

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "sample_case_report.html"
    generate_clinical_case_report(
        {"case_id": case_id, "patient_details": patient_details, "missing_inputs": []},
        summary,
        report_path,
        corridors,
        tumor_details=tumor_details,
        viewer_html=viewer_html,
        sample_note=sample_note,
        classification=classification,
    )
    shutil.copyfile(viewer_path, output_dir / "3d_case_viewer.html")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config_v7.json")
    parser.add_argument(
        "--summary", type=Path,
        default=PROJECT_ROOT / "outputs" / "V7_VALIDATED" / "patient_summary.json")
    parser.add_argument(
        "--corridors", type=Path,
        default=PROJECT_ROOT / "outputs" / "V8_CORRIDORS" / "corridor_paths.json")
    parser.add_argument(
        "--viewer", type=Path,
        default=PROJECT_ROOT / "outputs" / "V8_CORRIDORS" / "risk_3d_viewer.html")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sample_report")
    parser.add_argument("--case-id", default="UCSF-PDGM-0004")
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--sex", default=None)
    parser.add_argument("--history", default="")
    parser.add_argument(
        "--classification", default=None,
        help="Tumor classification supplied with the case, e.g. from pathology.")
    parser.add_argument(
        "--classification-source", default=None,
        help="Where the classification came from (required with --classification).")
    parser.add_argument("--classification-date", default=None)
    parser.add_argument(
        "--sample-note",
        default=(
            "This example uses a case from the public UCSF-PDGM research dataset to show the "
            "report format. Age, sex and history were not supplied with it."
        ),
    )
    args = parser.parse_args()
    classification = None
    if args.classification:
        if not args.classification_source:
            parser.error("--classification-source is required with --classification")
        classification = {
            "label": args.classification,
            "source": args.classification_source,
            "date": args.classification_date,
        }
    report = build_report(
        args.config.resolve(), args.summary.resolve(), args.corridors.resolve(),
        args.viewer.resolve(), args.output_dir.resolve(), args.case_id, args.age, args.sex,
        args.history, args.sample_note, classification,
    )
    print(f"Shareable report: {report}")


if __name__ == "__main__":
    main()
