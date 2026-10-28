"""Run the isolated V8 risk workflow for a dashboard-uploaded case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.clinical_case_report import generate_clinical_case_report
from pipeline.run_v8_pipeline import run_pipeline


def run_uploaded_case(config_path: Path, metadata_path: Path) -> Dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_pipeline(str(config_path))
    summary_path = Path(config["paths"]["output_dir_v7_validated"]) / "patient_summary.json"
    risk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    corridors_path = Path(config["paths"]["output_dir_v8_corridors"]) / "corridor_paths.json"
    corridors = (
        json.loads(corridors_path.read_text(encoding="utf-8")).get("corridors")
        if corridors_path.is_file() else None
    )
    report_path = Path(metadata["case_directory"]) / "medical_case_summary.html"
    record = generate_clinical_case_report(metadata, risk_summary, report_path, corridors)
    print(f"Research case summary: {report_path}")
    print("No compatible trained MRI model was run; risk mapping used the supplied masks.")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    run_uploaded_case(args.config.resolve(), args.metadata.resolve())


if __name__ == "__main__":
    main()
