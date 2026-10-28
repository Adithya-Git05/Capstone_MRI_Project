from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.config import load_config, resolve_path
from src.data.validation import validate_brats, validate_brisc, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BRISC and BraTS files and metadata")
    parser.add_argument("--all", action="store_true", help="Validate both datasets (the default)")
    parser.add_argument("--brisc-only", action="store_true")
    parser.add_argument("--brats-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.brisc_only and args.brats_only:
        parser.error("Choose at most one of --brisc-only and --brats-only")

    config = load_config("2_3d_segmentation/configs/segmentation3d.yaml")
    output_root = args.output_root or resolve_path(
        "1_data_and_preprocessing/outputs", env_var="OUTPUT_ROOT"
    )
    run_all = args.all or not (args.brisc_only or args.brats_only)
    reports = []
    if run_all or args.brisc_only:
        brisc_root = resolve_path(
            "1_data_and_preprocessing/data/BRISC/brisc2025", env_var="BRISC_ROOT"
        )
        report = validate_brisc(brisc_root)
        write_report(output_root / "data_validation" / "brisc.json", report)
        reports.append(report)
    if run_all or args.brats_only:
        brats_root = resolve_path(config["data_root"], env_var="BRATS_ROOT")
        dataset_config = resolve_path(config["dataset_config"])
        report = validate_brats(brats_root, dataset_config)
        write_report(output_root / "data_validation" / "brats.json", report)
        reports.append(report)

    print(json.dumps(
        [
            {
                "dataset": report["dataset"],
                "ok": report["ok"],
                "errors": report["errors"],
                "warnings": report["warnings"],
                "report_path": str(output_root / "data_validation" / (
                    "brisc.json" if report["dataset"].startswith("BRISC") else "brats.json"
                )),
            }
            for report in reports
        ],
        indent=2,
    ))
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
