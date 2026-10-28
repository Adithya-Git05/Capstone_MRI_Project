from __future__ import annotations

import json
import os
import shutil
import subprocess
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.common.config import PROJECT_ROOT, resolve_path
from src.data.brats import (
    DEFAULT_MODALITIES,
    DEFAULT_REGIONS,
    REGION_LABELS,
    REGION_ORDER,
    grouped_kfold,
    list_cases,
    load_full_volume,
    normalize_region_labels,
)
from src.segmentation3d.metrics import region_metrics

DATASET_ID = 501
DATASET_NAME = "Dataset501_BraTSAdditional"
NNUNET_MODALITY_ORDER = ("t2f", "t1n", "t1c", "t2w")


def _read_source_config(config: dict[str, Any]) -> dict[str, Any]:
    config_path = resolve_path(
        config.get(
            "dataset_config",
            "legacy/BraTS_Challenge/nnUNet_hyperparameters/dataset.json",
        )
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Source nnU-Net dataset config not found: {config_path}")
    source = json.loads(config_path.read_text(encoding="utf-8"))
    if not source.get("labels") or not source.get("channel_names"):
        raise ValueError(f"Source nnU-Net config lacks channels or labels: {config_path}")
    return source


def _assert_region_mappings(config: dict[str, Any], source_config: dict[str, Any]) -> None:
    if config.get("label_scheme") != "brats_post_treatment_composites_v1":
        raise ValueError(
            "The nnU-Net adapter requires the approved WT/TC/ET/RC ontology. "
            "Legacy L1-L4 configurations are not compatible."
        )
    if config.get("region_order") != list(REGION_ORDER):
        raise ValueError(f"region_order must be {list(REGION_ORDER)}")
    normalize_region_labels(config.get("region_labels", DEFAULT_REGIONS))
    source_labels = source_config["labels"]
    raw_label_values: set[int] = set()
    for value in source_labels.values():
        members = value if isinstance(value, (list, tuple)) else [value]
        raw_label_values.update(int(member) for member in members)
    if raw_label_values != {0, 1, 2, 3, 4}:
        raise ValueError(
            "Source nnU-Net label IDs do not match the raw-label ontology "
            f"(expected 0-4, got {sorted(raw_label_values)})"
        )
    source_channels = {
        int(index): str(name) for index, name in source_config["channel_names"].items()
    }
    expected_channels = {0: "FLAIR", 1: "T1", 2: "T1GD", 3: "T2"}
    if source_channels != expected_channels:
        raise ValueError(
            f"Unexpected source channel names/order: {source_channels}; "
            f"expected {expected_channels}"
        )


def _nnunet_region_labels() -> dict[str, int | list[int]]:
    return {
        "background": 0,
        **{name: list(REGION_LABELS[name]) for name in REGION_ORDER},
    }


def _nnunet_region_predictions(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 4 or probabilities.shape[0] != len(REGION_ORDER):
        raise ValueError(
            "nnU-Net region probabilities must have shape [4, X, Y, Z] "
            f"in channel order {REGION_ORDER}; received {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("nnU-Net region probabilities contain non-finite values")
    if probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError("nnU-Net region probabilities must be in the range [0, 1]")
    return probabilities >= 0.5


def _reorient_nnunet_region_predictions(
    probabilities: np.ndarray, affine: np.ndarray
) -> np.ndarray:
    masks = _nnunet_region_predictions(probabilities).astype(np.uint8)
    source_orientation = nib.orientations.io_orientation(affine)
    target_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(source_orientation, target_orientation)
    return np.stack(
        [nib.orientations.apply_orientation(region, transform) for region in masks],
        axis=0,
    ).astype(bool)


def _assert_generated_dataset_regions(dataset_dir: Path) -> None:
    config_path = dataset_dir / "dataset.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Generated nnU-Net dataset configuration is missing: {config_path}"
        )
    dataset_config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = dataset_config.get("labels")
    expected_names = ["background", *REGION_ORDER]
    expected_labels = _nnunet_region_labels()
    if (
        not isinstance(labels, dict)
        or list(labels) != expected_names
        or labels != expected_labels
        or dataset_config.get("region_order") != list(REGION_ORDER)
        or dataset_config.get("ontology_id") != "brats_post_treatment_composites_v1"
        or dataset_config.get("regions_class_order") != [1, 2, 3, 4]
    ):
        raise ValueError(
            "Generated nnU-Net dataset does not explicitly match the approved "
            f"channel order and mapping: expected {expected_names} and {expected_labels}"
        )


def prepare_nnunet_dataset(config: dict[str, Any], raw_root: Path) -> Path:
    data_root = resolve_path(config["data_root"], env_var="BRATS_ROOT")
    cases = list_cases(data_root)
    if len(cases) != len({case.case_id for case in cases}):
        raise ValueError("Duplicate BraTS case identifiers found")
    modalities = tuple(config.get("modalities", DEFAULT_MODALITIES))
    if modalities != NNUNET_MODALITY_ORDER:
        raise ValueError(
            f"nnU-Net adapter expects input order {NNUNET_MODALITY_ORDER}, got {modalities}"
        )
    source_config = _read_source_config(config)
    _assert_region_mappings(config, source_config)
    dataset_dir = raw_root / DATASET_NAME
    if dataset_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing nnU-Net dataset: {dataset_dir}; "
            "choose a new nnUNet work directory or move the existing output yourself."
        )
    metadata_path = raw_root.parent / "nnunet_experiment.json"
    if metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing nnU-Net experiment metadata: {metadata_path}"
        )
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    for case in cases:
        for channel, modality in enumerate(modalities):
            source = case.directory / f"{case.case_id}-{modality}.nii.gz"
            if not source.is_file():
                raise FileNotFoundError(f"Missing input modality: {source}")
            shutil.copy2(source, images_dir / f"{case.case_id}_{channel:04d}.nii.gz")
        segmentation = case.directory / f"{case.case_id}-seg.nii.gz"
        if not segmentation.is_file():
            raise FileNotFoundError(f"Missing segmentation: {segmentation}")
        shutil.copy2(segmentation, labels_dir / f"{case.case_id}.nii.gz")

    dataset_description = {
        "channel_names": source_config["channel_names"],
        "labels": _nnunet_region_labels(),
        "numTraining": len(cases),
        "file_ending": source_config.get("file_ending", ".nii.gz"),
        "regions_class_order": [1, 2, 3, 4],
        "region_order": list(REGION_ORDER),
        "ontology_id": "brats_post_treatment_composites_v1",
        "name": DATASET_NAME,
        "description": (
            "Local BraTS-format cases with approved overlapping WT/TC/ET/RC targets. "
            "Dataset release provenance is unverified."
        ),
        "reference": "BraTS post-treatment composite region definitions; local release unverified",
        "licence": "See original dataset terms",
        "release": "unverified local subset",
        "tensorImageSize": "4D",
        "target_representation": "overlapping binary regions",
        "raw_label_definitions": {
            name: list(REGION_LABELS[name]) for name in REGION_ORDER
        },
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(dataset_description, indent=2), encoding="utf-8"
    )
    package_versions = {}
    for package in ("nnunetv2", "torch", "numpy", "nibabel"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            continue
    metadata_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "package_versions": package_versions,
                "seed_for_grouped_splits": int(config.get("seed", 2026)),
                "case_count": len(cases),
                "patient_group_count": len({case.patient_group for case in cases}),
                "grouping_rule": "case folder prefix before final hyphen",
                "patient_group_identity_verified": False,
                "dataset_id": DATASET_ID,
                "dataset_name": DATASET_NAME,
                "source_data_root": str(data_root.resolve()),
                "resolved_config": {
                    key: value for key, value in config.items() if not key.startswith("_")
                },
                "region_order": list(REGION_ORDER),
                "ontology_id": "brats_post_treatment_composites_v1",
                "region_definitions": {
                    name: list(REGION_LABELS[name]) for name in REGION_ORDER
                },
                "target_representation": "overlapping binary regions",
                "release_provenance_verified": False,
                "model_training_folds": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_dir


def write_grouped_splits(
    cases_root: Path,
    preprocessed_dataset_dir: Path,
    fold_count: int,
    seed: int,
) -> Path:
    cases = list_cases(cases_root)
    splits = grouped_kfold(cases, fold_count, seed)
    output_path = preprocessed_dataset_dir / "splits_final.json"
    if not preprocessed_dataset_dir.is_dir():
        raise FileNotFoundError(
            f"nnU-Net preprocessing output is missing: {preprocessed_dataset_dir}"
        )
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing grouped splits: {output_path}; "
            "use a fresh work root to change the seed or fold count."
        )
    output_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return output_path


def nnunet_environment(work_root: Path) -> dict[str, str]:
    nnunet_raw = work_root / "nnUNet_raw"
    preprocessed = work_root / "nnUNet_preprocessed"
    results = work_root / "nnUNet_results"
    for path in (nnunet_raw, preprocessed, results):
        path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "nnUNet_raw": str(nnunet_raw.resolve()),
            "nnUNet_preprocessed": str(preprocessed.resolve()),
            "nnUNet_results": str(results.resolve()),
        }
    )
    return environment


def plan_and_preprocess(config: dict[str, Any], work_root: Path) -> Path:
    preprocessed_dataset_dir = work_root / "nnUNet_preprocessed" / DATASET_NAME
    if preprocessed_dataset_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite preprocessed nnU-Net outputs: "
            f"{preprocessed_dataset_dir}; use a fresh work root."
        )
    _assert_generated_dataset_regions(
        work_root / "nnUNet_raw" / DATASET_NAME
    )
    environment = nnunet_environment(work_root)
    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(DATASET_ID),
        "--verify_dataset_integrity",
    ]
    subprocess.run(command, check=True, env=environment, cwd=PROJECT_ROOT)
    preprocessed_dir = Path(environment["nnUNet_preprocessed"]) / DATASET_NAME
    splits = write_grouped_splits(
        resolve_path(config["data_root"], env_var="BRATS_ROOT"),
        preprocessed_dir,
        int(config.get("folds", 5)),
        int(config.get("seed", 2026)),
    )
    metadata_path = work_root / "nnunet_experiment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["grouped_splits_file"] = str(splits.resolve())
    metadata["fold_count"] = int(config.get("folds", 5))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return preprocessed_dir


def train_nnunet(work_root: Path, fold: int) -> None:
    environment = nnunet_environment(work_root)
    _assert_generated_dataset_regions(
        Path(environment["nnUNet_raw"]) / DATASET_NAME
    )
    splits_path = Path(environment["nnUNet_preprocessed"]) / DATASET_NAME / "splits_final.json"
    if not splits_path.is_file():
        raise FileNotFoundError(
            "Grouped splits are not prepared; run the preprocess command first."
        )
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if not 0 <= fold < len(splits):
        raise ValueError(f"fold must be in [0, {len(splits) - 1}]")
    results_root = Path(environment["nnUNet_results"])
    if list(results_root.rglob(f"fold_{fold}")):
        raise FileExistsError(
            f"Refusing to retrain or overwrite existing nnU-Net fold {fold} in "
            f"{results_root}; use a fresh work root."
        )
    subprocess.run(
        [
            "nnUNetv2_train",
            DATASET_NAME,
            "3d_fullres",
            str(fold),
            "--npz",
        ],
        check=True,
        env=environment,
        cwd=PROJECT_ROOT,
    )
    metadata_path = work_root / "nnunet_experiment.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"nnU-Net experiment metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trained_folds = set(metadata.get("model_training_folds", []))
    trained_folds.add(fold)
    metadata["model_training_folds"] = sorted(trained_folds)
    metadata["last_trained_at"] = datetime.now().astimezone().isoformat()
    fold_directories = sorted(results_root.rglob(f"fold_{fold}"))
    metadata.setdefault("fold_results", {})[str(fold)] = {
        "directories": [str(path.resolve()) for path in fold_directories],
        "checkpoints": [
            str(checkpoint.resolve())
            for fold_directory in fold_directories
            for checkpoint in fold_directory.glob("checkpoint_*.pth")
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def evaluate_nnunet_validation(
    config: dict[str, Any],
    work_root: Path,
    fold: int,
    output_dir: Path,
) -> Path:
    from src.segmentation3d.monai_engine import (
        _ontology_metadata,
        _require_new_or_empty_output_dir,
        _save_evaluation,
        _save_prediction_nifti,
        _summarize,
    )

    _require_new_or_empty_output_dir(output_dir)
    environment = nnunet_environment(work_root)
    _assert_generated_dataset_regions(
        Path(environment["nnUNet_raw"]) / DATASET_NAME
    )
    preprocessed_dir = Path(environment["nnUNet_preprocessed"]) / DATASET_NAME
    experiment_metadata_path = work_root / "nnunet_experiment.json"
    if not experiment_metadata_path.is_file():
        raise FileNotFoundError(
            f"nnU-Net experiment metadata is missing: {experiment_metadata_path}"
        )
    experiment_metadata = json.loads(
        experiment_metadata_path.read_text(encoding="utf-8")
    )
    splits_path = preprocessed_dir / "splits_final.json"
    if not splits_path.is_file():
        raise FileNotFoundError(f"Grouped fold definitions are missing: {splits_path}")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if not 0 <= fold < len(splits):
        raise ValueError(f"fold must be in [0, {len(splits) - 1}]")
    validation_ids = set(splits[fold]["val"])
    cases = {case.case_id: case for case in list_cases(
        resolve_path(config["data_root"], env_var="BRATS_ROOT")
    )}
    if validation_ids - cases.keys():
        raise ValueError("Grouped split references cases absent from the configured BraTS root")

    validation_input = work_root / "validation_inputs" / f"fold_{fold}"
    if validation_input.exists():
        raise FileExistsError(
            f"Refusing to overwrite validation inputs: {validation_input}; use a new work root."
        )
    validation_input.mkdir(parents=True)
    raw_images = Path(environment["nnUNet_raw"]) / DATASET_NAME / "imagesTr"
    for case_id in sorted(validation_ids):
        for channel in range(4):
            source = raw_images / f"{case_id}_{channel:04d}.nii.gz"
            if not source.is_file():
                raise FileNotFoundError(f"Prepared nnU-Net input is missing: {source}")
            shutil.copy2(source, validation_input / source.name)

    prediction_dir = output_dir / "predictions"
    if prediction_dir.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {prediction_dir}")
    prediction_dir.mkdir(parents=True)
    subprocess.run(
        [
            "nnUNetv2_predict",
            "-d",
            DATASET_NAME,
            "-i",
            str(validation_input.resolve()),
            "-o",
            str(prediction_dir.resolve()),
            "-f",
            str(fold),
            "-c",
            "3d_fullres",
            "--save_probabilities",
        ],
        check=True,
        env=environment,
        cwd=PROJECT_ROOT,
    )

    source_config = _read_source_config(config)
    _assert_region_mappings(config, source_config)
    region_names = list(REGION_ORDER)
    region_labels = {
        name: list(REGION_LABELS[name]) for name in REGION_ORDER
    }
    rows = []
    visualizations = []
    for case_id in sorted(validation_ids):
        prediction_path = prediction_dir / f"{case_id}.nii.gz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"nnU-Net prediction is missing: {prediction_path}")
        source_prediction_image = nib.load(str(prediction_path))
        predicted_image = nib.as_closest_canonical(source_prediction_image)
        probability_path = prediction_dir / f"{case_id}.npz"
        if not probability_path.is_file():
            raise FileNotFoundError(
                f"nnU-Net region probabilities are missing: {probability_path}. "
                "Evaluation requires --save_probabilities to preserve overlapping outputs."
            )
        with np.load(probability_path) as probability_file:
            if "probabilities" not in probability_file:
                raise ValueError(
                    f"nnU-Net probability archive lacks 'probabilities': {probability_path}"
                )
            probabilities = probability_file["probabilities"]
        predictions = _reorient_nnunet_region_predictions(
            probabilities, source_prediction_image.affine
        )
        volume = load_full_volume(
            cases[case_id],
            modalities=tuple(config.get("modalities", DEFAULT_MODALITIES)),
            orientation=str(config.get("orientation", "RAS")),
            region_labels=region_labels,
        )
        targets = volume["label"].numpy() >= 0.5
        if predicted_image.shape != targets.shape or not np.allclose(
            predicted_image.affine, volume["affine"], atol=1e-3
        ):
            raise ValueError(f"Prediction geometry does not match reference for {case_id}")
        if predictions.shape != targets.shape:
            raise ValueError(
                f"nnU-Net probabilities do not match target shape for {case_id}: "
                f"{predictions.shape} != {targets.shape}"
            )
        _save_prediction_nifti(
            output_dir / "region_masks",
            case_id,
            predictions,
            region_names,
            volume["affine"],
        )
        case_metrics = region_metrics(
            predictions, targets, region_names, volume["spacing"]
        )
        for region_name, metrics in case_metrics.items():
            rows.append({"case_id": case_id, "patient_group": cases[case_id].patient_group,
                         "region": region_name, **metrics})
        if len(visualizations) < 4:
            image_channel = tuple(config.get("modalities", DEFAULT_MODALITIES)).index("t1c")
            visualizations.append(
                {
                    "case_id": case_id,
                    "image": volume["image"][image_channel].numpy(),
                    "target": targets,
                    "prediction": predictions,
                }
            )

    _save_evaluation(
        output_dir,
        rows,
        _summarize(rows, f"grouped_validation_fold_{fold}", None, region_labels),
        visualizations,
    )
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "pipeline": "brats_3d_nnunet",
                "fold": fold,
                "validation_case_count": len(validation_ids),
                "seed": int(config.get("seed", 2026)),
                "training_case_count": len(splits[fold]["train"]),
                "training_case_ids": sorted(splits[fold]["train"]),
                "validation_case_ids": sorted(validation_ids),
                "experiment_metadata_file": str(experiment_metadata_path.resolve()),
                "python": experiment_metadata.get("python"),
                "platform": experiment_metadata.get("platform"),
                "package_versions": experiment_metadata.get("package_versions"),
                "resolved_config": {
                    key: value for key, value in config.items() if not key.startswith("_")
                },
                "checkpoint_dir": environment["nnUNet_results"],
                "prediction_dir": str(prediction_dir.resolve()),
                "source_data_root": str(resolve_path(
                    config["data_root"], env_var="BRATS_ROOT"
                )),
                "official_test_split": False,
                "region_order": list(REGION_ORDER),
                "region_definitions": {
                    name: list(REGION_LABELS[name]) for name in REGION_ORDER
                },
                "target_representation": "overlapping binary regions",
                "prediction_threshold": 0.5,
                "ontology": _ontology_metadata(),
                "release_provenance_verified": False,
                "device": "selected by nnU-Net runtime; not independently captured",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    import argparse

    from src.common.config import load_config

    parser = argparse.ArgumentParser(description="Prepare or train nnU-Net on local BraTS cases")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", default="2_3d_segmentation/configs/segmentation3d.yaml")
    prepare_parser.add_argument("--work-root", type=Path, default=Path("2_3d_segmentation/outputs/segmentation3d/nnunet_work"))
    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--config", default="2_3d_segmentation/configs/segmentation3d.yaml")
    preprocess_parser.add_argument("--work-root", type=Path, default=Path("2_3d_segmentation/outputs/segmentation3d/nnunet_work"))
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--work-root", type=Path, default=Path("2_3d_segmentation/outputs/segmentation3d/nnunet_work"))
    train_parser.add_argument("--fold", type=int, default=0)
    evaluate_parser = subparsers.add_parser("evaluate-validation")
    evaluate_parser.add_argument("--config", default="2_3d_segmentation/configs/segmentation3d.yaml")
    evaluate_parser.add_argument("--work-root", type=Path, default=Path("2_3d_segmentation/outputs/segmentation3d/nnunet_work"))
    evaluate_parser.add_argument("--fold", type=int, default=0)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(getattr(args, "config", "2_3d_segmentation/configs/segmentation3d.yaml"))
    work_root = resolve_path(args.work_root)
    if args.command == "prepare":
        dataset_dir = prepare_nnunet_dataset(config, work_root / "nnUNet_raw")
        print(dataset_dir)
    elif args.command == "preprocess":
        print(plan_and_preprocess(config, work_root))
    elif args.command == "train":
        train_nnunet(work_root, args.fold)
    else:
        print(evaluate_nnunet_validation(
            config, work_root, args.fold, resolve_path(args.output_dir)
        ))
    return 0
