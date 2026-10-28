from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.brisc import CLASS_NAMES

BRATS_MODALITIES = ("t1n", "t1c", "t2w", "t2f", "seg")


def validate_brisc(data_root: Path) -> dict[str, Any]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"BRISC data root does not exist: {data_root}")
    report: dict[str, Any] = {
        "dataset": "BRISC 2025",
        "data_root": str(data_root.resolve()),
        "errors": [],
        "warnings": [],
        "classification": {},
        "segmentation": {},
        "manifest": {},
    }

    for split in ("train", "test"):
        split_root = data_root / "classification_task" / split
        class_counts: dict[str, int] = {}
        for class_name in CLASS_NAMES:
            folder = split_root / class_name
            if not folder.is_dir():
                report["errors"].append(f"Missing classification folder: {folder}")
                class_counts[class_name] = 0
                continue
            files = sorted(p for p in folder.iterdir() if p.is_file())
            class_counts[class_name] = len(files)
            for path in files:
                _verify_image(path, report["errors"])
        report["classification"][split] = {
            "class_counts": class_counts,
            "sample_count": sum(class_counts.values()),
        }

        seg_root = data_root / "segmentation_task" / split
        image_dir, mask_dir = seg_root / "images", seg_root / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            report["errors"].append(f"Missing segmentation image or mask folder: {seg_root}")
            continue
        images = {p.stem: p for p in image_dir.iterdir() if p.is_file()}
        masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}
        missing_masks = sorted(images.keys() - masks.keys())
        orphan_masks = sorted(masks.keys() - images.keys())
        if missing_masks:
            report["errors"].append(f"{split}: {len(missing_masks)} images have no mask")
        if orphan_masks:
            report["errors"].append(f"{split}: {len(orphan_masks)} masks have no image")

        size_mismatches = []
        mask_values: set[int] = set()
        mask_pixel_counts: Counter[int] = Counter()
        image_sizes: Counter[str] = Counter()
        for stem in sorted(images.keys() & masks.keys()):
            image_path, mask_path = images[stem], masks[stem]
            try:
                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    image.load()
                    mask.load()
                    if image.size != mask.size:
                        size_mismatches.append(stem)
                    image_sizes[f"{image.width}x{image.height}"] += 1
                    if mask.mode not in ("L", "1", "P"):
                        report["errors"].append(f"Unexpected mask mode {mask.mode}: {mask_path}")
                    values, counts = np.unique(np.asarray(mask.convert("L")), return_counts=True)
                    mask_values.update(int(value) for value in values)
                    mask_pixel_counts.update(
                        {int(value): int(count) for value, count in zip(values, counts)}
                    )
            except (OSError, ValueError) as error:
                report["errors"].append(f"Could not read pair {stem}: {error}")
        if size_mismatches:
            report["errors"].append(f"{split}: {len(size_mismatches)} image/mask dimension mismatches")
        unexpected_mask_values = sorted(value for value in mask_values if not 0 <= value <= 255)
        if unexpected_mask_values:
            report["errors"].append(f"{split}: mask values outside 8-bit range")
        report["segmentation"][split] = {
            "image_count": len(images),
            "mask_count": len(masks),
            "paired_count": len(images.keys() & masks.keys()),
            "unmatched_images": len(missing_masks),
            "orphan_masks": len(orphan_masks),
            "dimension_mismatches": len(size_mismatches),
            "image_size_counts": dict(image_sizes),
            "raw_mask_values": sorted(mask_values),
            "threshold": 128,
            "thresholded_mask_values": sorted(
                {int(value >= 128) for value in mask_values}
            ),
            "foreground_fraction": float(
                sum(count for value, count in mask_pixel_counts.items() if value >= 128)
                / max(1, sum(mask_pixel_counts.values()))
            ),
        }

    manifest_path = data_root / "manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        report["manifest"]["csv_rows"] = len(rows)
        missing_manifest_paths = []
        for row in rows:
            relative_path = row.get("relative_path", "").replace("\\", "/")
            if not relative_path or not (data_root / Path(relative_path)).is_file():
                missing_manifest_paths.append(relative_path)
        report["manifest"]["missing_paths"] = len(missing_manifest_paths)
        if missing_manifest_paths:
            report["errors"].append(
                f"Manifest references {len(missing_manifest_paths)} missing files"
            )
        report["manifest"]["file_count"] = sum(
            1
            for path in data_root.rglob("*")
            if path.is_file() and path.name not in {"manifest.csv", "manifest.json"}
            and not path.name.endswith(".sha256")
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        report["manifest"]["row_count_matches_files"] = (
            report["manifest"]["csv_rows"] == report["manifest"]["file_count"]
        )
        if not report["manifest"]["row_count_matches_files"]:
            report["warnings"].append("Manifest row count differs from the on-disk image/mask count")
    _check_checksum(data_root / "manifest.csv", report["warnings"])
    _check_checksum(data_root / "manifest.json", report["warnings"])
    report["patient_level_split_integrity"] = (
        "unverifiable: BRISC filenames/manifests do not expose a patient identifier"
    )
    report["ok"] = not report["errors"]
    return report


def _verify_image(path: Path, errors: list[str]) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.width < 1 or image.height < 1:
                errors.append(f"Invalid image dimensions: {path}")
    except (OSError, ValueError) as error:
        errors.append(f"Could not read image {path}: {error}")


def _check_checksum(path: Path, warnings: list[str]) -> None:
    checksum_path = path.with_name(path.name + ".sha256")
    if not path.is_file() or not checksum_path.is_file():
        warnings.append(f"Checksum metadata missing for {path.name}")
        return
    expected = checksum_path.read_text(encoding="ascii").strip().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected.lower() != actual.lower():
        warnings.append(f"SHA-256 mismatch for {path.name}")


def validate_brats(
    data_root: Path,
    dataset_config_path: Path | None = None,
) -> dict[str, Any]:
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError("BraTS validation requires nibabel; install the [brats] extra") from error

    if not data_root.is_dir():
        raise FileNotFoundError(f"BraTS data root does not exist: {data_root}")
    cases = sorted(path for path in data_root.iterdir() if path.is_dir())
    report: dict[str, Any] = {
        "dataset": "BraTS-GLI local additional training cases",
        "data_root": str(data_root.resolve()),
        "case_count": len(cases),
        "errors": [],
        "warnings": [],
        "missing_modalities": [],
        "shape_mismatches": [],
        "affine_mismatches": [],
        "spacing_counts": Counter(),
        "orientation_counts": Counter(),
        "shape_counts": Counter(),
        "label_value_counts": Counter(),
        "patient_group_count": 0,
        "patient_groups_with_multiple_cases": 0,
    }
    groups: dict[str, list[str]] = defaultdict(list)
    all_labels: set[int] = set()
    for case_dir in cases:
        case_id = case_dir.name
        groups[case_id.rsplit("-", 1)[0]].append(case_id)
        images = {}
        for modality in BRATS_MODALITIES:
            path = case_dir / f"{case_id}-{modality}.nii.gz"
            if not path.is_file():
                report["missing_modalities"].append(f"{case_id}:{modality}")
                continue
            try:
                images[modality] = nib.load(str(path))
            except (OSError, ValueError) as error:
                report["errors"].append(f"Unreadable NIfTI {path}: {error}")
        if len(images) != len(BRATS_MODALITIES):
            continue
        reference = images["seg"]
        reference_affine = reference.affine
        shape = tuple(int(value) for value in reference.shape)
        report["shape_counts"][str(shape)] += 1
        for modality, image in images.items():
            if image.shape != reference.shape:
                report["shape_mismatches"].append(f"{case_id}:{modality}")
            if not np.allclose(image.affine, reference_affine, atol=1e-4):
                report["affine_mismatches"].append(f"{case_id}:{modality}")
            report["spacing_counts"][str(tuple(round(float(v), 5) for v in image.header.get_zooms()[:3]))] += 1
            report["orientation_counts"][str(nib.aff2axcodes(image.affine))] += 1
        try:
            labels = np.unique(np.asanyarray(reference.dataobj))
            int_labels = {int(value) for value in labels}
            all_labels.update(int_labels)
            report["label_value_counts"][str(tuple(sorted(int_labels)))] += 1
            if not int_labels.issubset({0, 1, 2, 3, 4}):
                report["errors"].append(f"Unexpected label value in {case_id}: {sorted(int_labels)}")
        except (OSError, ValueError) as error:
            report["errors"].append(f"Could not read segmentation labels for {case_id}: {error}")

    if report["missing_modalities"]:
        report["errors"].append(
            f"{len(report['missing_modalities'])} expected modality/label files are missing"
        )
    if report["shape_mismatches"]:
        report["errors"].append(
            f"{len(report['shape_mismatches'])} modality shape mismatches found"
        )
    if report["affine_mismatches"]:
        report["errors"].append(
            f"{len(report['affine_mismatches'])} modality affine mismatches found"
        )
    report["patient_group_count"] = len(groups)
    report["patient_groups_with_multiple_cases"] = sum(len(items) > 1 for items in groups.values())
    report["shared_prefix_grouping"] = (
        "Cases sharing the filename prefix before the last hyphen are kept together; "
        "this is a conservative grouping inferred from case IDs, not an independently verified patient registry."
    )
    report["label_values_observed"] = sorted(all_labels)
    report["num_training_declared_by_config"] = None
    if dataset_config_path is not None and dataset_config_path.is_file():
        dataset_config = json.loads(dataset_config_path.read_text(encoding="utf-8"))
        declared = dataset_config.get("numTraining")
        report["num_training_declared_by_config"] = declared
        if declared != len(cases):
            report["warnings"].append(
                f"Config declares numTraining={declared}; local folders contain {len(cases)} cases. "
                "The config is not silently rewritten."
            )
        report["configured_channels"] = dataset_config.get("channel_names")
        report["configured_labels"] = dataset_config.get("labels")
    report["ok"] = not report["errors"]
    report["spacing_counts"] = dict(report["spacing_counts"])
    report["orientation_counts"] = dict(report["orientation_counts"])
    report["shape_counts"] = dict(report["shape_counts"])
    report["label_value_counts"] = dict(report["label_value_counts"])
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
