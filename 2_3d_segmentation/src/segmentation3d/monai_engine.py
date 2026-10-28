from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.config import load_config, resolve_path, runtime_device
from src.common.reproducibility import seed_everything, worker_seed
from src.common.run import (
    configure_logging,
    create_run_dir,
    update_run_metadata,
    write_run_metadata,
)
from src.data.brats import (
    DEFAULT_MODALITIES,
    DEFAULT_REGIONS,
    REGION_LABELS,
    REGION_ORDER,
    BraTSCase,
    BraTSVolumeDataset,
    grouped_train_validation_split,
    list_cases,
    load_full_volume,
    normalize_region_labels,
)
from src.segmentation3d.metrics import region_metrics
from src.segmentation3d.models import build_monai_model


ONTOLOGY_ID = "brats_post_treatment_composites_v1"


def _ontology_metadata() -> dict[str, Any]:
    return {
        "ontology_id": ONTOLOGY_ID,
        "channel_order": list(REGION_ORDER),
        "raw_label_definitions": {
            name: list(REGION_LABELS[name]) for name in REGION_ORDER
        },
        "target_representation": "overlapping_binary_masks",
        "release_provenance_verified": False,
    }


def _regions(config: dict[str, Any]) -> dict[str, list[int]]:
    label_scheme = config.get("label_scheme")
    if label_scheme != ONTOLOGY_ID:
        raise ValueError(
            f"Unsupported label_scheme {label_scheme!r}; expected {ONTOLOGY_ID!r}. "
            "Legacy L1-L4 configurations must not be reinterpreted as WT/TC/ET/RC."
        )
    configured_order = config.get("region_order")
    if configured_order != list(REGION_ORDER):
        raise ValueError(
            f"region_order must be {list(REGION_ORDER)}; received {configured_order!r}"
        )
    normalized = normalize_region_labels(config.get("region_labels", DEFAULT_REGIONS))
    return {name: list(normalized[name]) for name in REGION_ORDER}


def _validate_checkpoint_ontology(checkpoint: dict[str, Any]) -> None:
    ontology = checkpoint.get("ontology")
    if ontology is None:
        raise ValueError(
            "Checkpoint has no verified 3D segmentation ontology metadata. "
            "Legacy checkpoints cannot be loaded as WT/TC/ET/RC; retrain with "
            "the approved ontology or use the checkpoint only with its historical protocol."
        )
    if ontology != _ontology_metadata():
        raise ValueError(
            "Checkpoint ontology metadata does not match the approved WT/TC/ET/RC "
            "overlapping-mask ontology; refusing to load it."
        )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing its resolved configuration.")
    _regions(config)


def _load_compatible_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid 3D checkpoint format: {checkpoint_path}")
    _validate_checkpoint_ontology(checkpoint)
    return checkpoint


def _require_new_or_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing experiment/evaluation outputs in {output_dir}. "
            "Choose a new output directory."
        )


def _modalities(config: dict[str, Any]) -> tuple[str, ...]:
    modalities = tuple(config.get("modalities", DEFAULT_MODALITIES))
    if len(modalities) != 4 or len(set(modalities)) != 4:
        raise ValueError("BraTS requires four unique MRI modalities")
    return modalities


def _model(config: dict[str, Any], model_name: str):
    try:
        import monai
    except ImportError as error:
        raise RuntimeError("MONAI training requires `pip install -e .[brats]`") from error
    del monai
    model_config = config.get("model", {})
    feature_key = {
        "swinunetr": "swinunetr_feature_size",
        "resnet3d": "resnet3d_feature_size",
        "efficientnet3d": "efficientnet3d_feature_size",
    }.get(model_name, "segresnet_feature_size")
    return build_monai_model(
        model_name,
        patch_size=config.get("patch_size", [96, 96, 96]),
        in_channels=4,
        out_channels=len(REGION_ORDER),
        feature_size=int(model_config.get(feature_key, 48 if model_name == "swinunetr" else 16)),
    )


def _loader(
    dataset: BraTSVolumeDataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=worker_seed,
        pin_memory=torch.cuda.is_available(),
    )


def _soft_dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)
    spatial_dims = tuple(range(2, logits.ndim))
    intersection = (probabilities * target).sum(dim=spatial_dims)
    denominator = probabilities.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


def _predict_logits(
    model, image: torch.Tensor, patch_size: tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    from monai.inferers import sliding_window_inference

    model.eval()
    with torch.inference_mode():
        prediction = sliding_window_inference(
            inputs=image.unsqueeze(0).to(device),
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )[0]
    return prediction


def _evaluate_cases(
    model,
    cases: list[BraTSCase],
    config: dict[str, Any],
    device: torch.device,
    *,
    output_dir: Path | None = None,
    compute_hd95: bool = True,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    regions = _regions(config)
    region_names = list(REGION_ORDER)
    patch_size = tuple(int(value) for value in config.get("patch_size", [96, 96, 96]))
    modalities = _modalities(config)
    rows: list[dict[str, Any]] = []
    sample_visualizations: list[dict[str, Any]] = []
    losses = []
    model.eval()
    for case in cases:
        volume = load_full_volume(
            case,
            modalities=modalities,
            orientation=str(config.get("orientation", "RAS")),
            region_labels=regions,
        )
        image = volume["image"]
        target = volume["label"].numpy() >= 0.5
        logits = _predict_logits(model, image, patch_size, device)
        losses.append(
            float(
                _soft_dice_bce(
                    logits.unsqueeze(0), volume["label"].unsqueeze(0).to(logits.device)
                ).item()
            )
        )
        prediction = torch.sigmoid(logits).cpu().numpy() >= 0.5
        metrics = region_metrics(
            prediction, target, region_names, volume["spacing"], compute_hd95=compute_hd95
        )
        for region_name, values in metrics.items():
            rows.append({"case_id": case.case_id, "patient_group": case.patient_group,
                         "region": region_name, **values})
        if output_dir is not None:
            _save_prediction_nifti(output_dir, case.case_id, prediction, region_names, volume["affine"])
        if len(sample_visualizations) < 4:
            image_channel = modalities.index("t1c") if "t1c" in modalities else 0
            sample_visualizations.append(
                {
                    "case_id": case.case_id,
                    "image": image[image_channel].numpy(),
                    "target": target,
                    "prediction": prediction,
                }
            )
    if not rows:
        raise ValueError("Cannot evaluate an empty BraTS split")
    return float(np.mean(losses)), rows, sample_visualizations


def _summarize(
    rows: list[dict[str, Any]],
    split: str,
    loss: float | None,
    region_definitions: dict[str, list[int]],
) -> dict[str, Any]:
    region_names = list(REGION_ORDER)
    observed_names = {row["region"] for row in rows}
    if observed_names != set(region_names):
        raise ValueError(
            f"Metric rows must contain exactly {region_names}; received {sorted(observed_names)}"
        )
    per_region = {}
    for region_name in region_names:
        subset = [row for row in rows if row["region"] == region_name]
        valid_hd95 = [row["hd95_mm"] for row in subset if row["hd95_mm"] is not None]
        per_region[region_name] = {
            "case_count": len(subset),
            "mean_dice": float(np.mean([row["dice"] for row in subset])),
            "mean_iou": float(np.mean([row["iou"] for row in subset])),
            "mean_hd95_mm_when_valid": float(np.mean(valid_hd95)) if valid_hd95 else None,
            "hd95_valid_case_count": len(valid_hd95),
        }
    return {
        "split": split,
        "case_count": len({row["case_id"] for row in rows}),
        "region_count": len(region_names),
        "bce_plus_soft_dice_loss": loss,
        "per_region": per_region,
        "region_order": list(REGION_ORDER),
        "region_definitions": {
            name: region_definitions[name] for name in REGION_ORDER
        },
        "prediction_threshold": 0.5,
        "ontology": _ontology_metadata(),
    }


def _save_prediction_nifti(
    output_dir: Path,
    case_id: str,
    prediction: np.ndarray,
    region_names: list[str],
    affine: np.ndarray,
) -> None:
    import nibabel as nib

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, region_name in enumerate(region_names):
        output_path = output_dir / f"{case_id}_{region_name}.nii.gz"
        image = nib.Nifti1Image(prediction[index].astype(np.uint8), affine)
        nib.save(image, str(output_path))


def _save_figures(
    output_dir: Path,
    rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    history: list[dict[str, float]] | None = None,
) -> None:
    if history:
        epochs = [item["epoch"] for item in history]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
        axes[0].plot(epochs, [item["validation_loss"] for item in history], label="validation")
        axes[0].set(xlabel="Epoch", ylabel="BCE + soft Dice", title="Loss")
        axes[0].legend()
        axes[1].plot(epochs, [item["validation_dice"] for item in history], label="validation")
        axes[1].set(
            xlabel="Epoch",
            ylabel="Mean WT/TC/ET/RC Dice",
            ylim=(0, 1),
            title="Validation",
        )
        axes[1].legend()
        figure.savefig(output_dir / "training_curves.png", dpi=300)
        figure.savefig(output_dir / "training_curves.pdf")
        plt.close(figure)
        with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    if samples:
        figure, axes = plt.subplots(len(samples), 3, figsize=(10, 3.5 * len(samples)), squeeze=False)
        for index, sample in enumerate(samples):
            tumor = sample["target"].any(axis=0)
            if tumor.any():
                z_index = int(np.argmax(tumor.sum(axis=(0, 1))))
            else:
                z_index = sample["image"].shape[-1] // 2
            image = sample["image"][:, :, z_index]
            low, high = np.percentile(image, (1, 99))
            display = np.clip((image - low) / max(float(high - low), 1e-8), 0, 1)
            target = sample["target"][:, :, :, z_index].any(axis=0)
            prediction = sample["prediction"][:, :, :, z_index].any(axis=0)
            axes[index, 0].imshow(display, cmap="gray")
            axes[index, 0].set_title(f"{sample['case_id']} MRI")
            axes[index, 1].imshow(display, cmap="gray")
            axes[index, 1].imshow(target, cmap="Reds", alpha=0.45)
            axes[index, 1].set_title("Ground-truth union overlay (WT/TC/ET/RC)")
            axes[index, 2].imshow(display, cmap="gray")
            axes[index, 2].imshow(prediction, cmap="Blues", alpha=0.45)
            axes[index, 2].set_title("Prediction union overlay (WT/TC/ET/RC)")
            for axis in axes[index]:
                axis.axis("off")
        figure.savefig(output_dir / "representative_3d_slices.png", dpi=300)
        figure.savefig(output_dir / "representative_3d_slices.pdf")
        plt.close(figure)


def _save_evaluation(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
    history: list[dict[str, float]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "per_case_region_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _save_figures(output_dir, rows, samples, history)


def train(config: dict[str, Any], model_name: str) -> Path:
    seed = int(config.get("seed", 2026))
    seed_everything(seed)
    data_root = resolve_path(config["data_root"], env_var="BRATS_ROOT")
    output_root = resolve_path(config["output_root"], env_var="OUTPUT_ROOT") / model_name
    cases = list_cases(data_root)
    training_cases, validation_cases = grouped_train_validation_split(
        cases, float(config.get("validation_fraction", 0.2)), seed
    )
    device = torch.device(runtime_device(str(config.get("device", "auto"))))
    model = _model(config, model_name).to(device)
    patch_size = tuple(int(value) for value in config.get("patch_size", [96, 96, 96]))
    modalities = _modalities(config)
    train_dataset = BraTSVolumeDataset(
        training_cases,
        patch_size=patch_size,
        modalities=modalities,
        orientation=str(config.get("orientation", "RAS")),
        augment=True,
        region_labels=_regions(config),
    )
    batch_size = int(config.get("batch_size", 1))
    train_loader = _loader(
        train_dataset, batch_size, seed, True, int(config.get("num_workers", 0))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.0001)),
        weight_decay=float(config.get("weight_decay", 0.00001)),
    )
    run_dir = create_run_dir(output_root, f"seed{seed}")
    logger = configure_logging(run_dir)
    validation_groups = sorted({case.patient_group for case in validation_cases})
    training_groups = sorted({case.patient_group for case in training_cases})
    if set(training_groups) & set(validation_groups):
        raise RuntimeError("Patient-group leakage detected in BraTS train/validation split")
    write_run_metadata(
        run_dir,
        config,
        pipeline=f"brats_3d_{model_name}",
        split_details={
            "total_cases": len(cases),
            "training_cases": len(training_cases),
            "validation_cases": len(validation_cases),
            "training_group_count": len(training_groups),
            "validation_group_count": len(validation_groups),
            "training_case_ids": sorted(case.case_id for case in training_cases),
            "validation_case_ids": sorted(case.case_id for case in validation_cases),
            "training_group_ids": training_groups,
            "validation_group_ids": validation_groups,
            "grouping_rule": "case folder prefix before final hyphen",
            "group_identity_verified": False,
            "official_test_split_available": False,
            "modalities_in_input_order": list(modalities),
            "region_order": list(REGION_ORDER),
            "region_definitions": _regions(config),
            "ontology": _ontology_metadata(),
            "checkpoint_selection_criterion": (
                "maximum mean per-case Dice across WT, TC, ET, and RC on grouped validation"
            ),
        },
    )
    update_run_metadata(run_dir, device=str(device), data_root=str(data_root.resolve()))
    history: list[dict[str, float]] = []
    best_dice = -1.0
    best_epoch = 0
    stale_epochs = 0
    patience = int(config.get("early_stopping_patience", 30))
    for epoch in range(1, int(config.get("epochs", 300)) + 1):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _soft_dice_bce(model(image), target)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * image.size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)
        validation_loss, validation_rows, _ = _evaluate_cases(
            model, validation_cases, config, device, compute_hd95=False
        )
        validation_dice = float(np.mean([row["dice"] for row in validation_rows]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_dice": validation_dice,
            }
        )
        logger.info(
            "epoch=%d train_loss=%.5f validation_loss=%.5f validation_mean_region_dice=%.4f",
            epoch, train_loss, validation_loss, validation_dice,
        )
        if validation_dice > best_dice:
            best_dice = validation_dice
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "config": {key: value for key, value in config.items() if not key.startswith("_")},
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "checkpoint_selection_criterion": (
                        "maximum mean per-case Dice across WT, TC, ET, and RC on grouped validation"
                    ),
                    "ontology": _ontology_metadata(),
                },
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("early stopping at epoch %d; best epoch=%d", epoch, best_epoch)
                break
    checkpoint = _load_compatible_checkpoint(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    validation_loss, rows, samples = _evaluate_cases(
        model, validation_cases, config, device, output_dir=run_dir / "region_masks"
    )
    summary = _summarize(rows, "grouped_validation", validation_loss, _regions(config))
    _save_evaluation(run_dir, rows, summary, samples, history)
    update_run_metadata(
        run_dir,
        checkpoint=str((run_dir / "best.pt").resolve()),
        metrics=str((run_dir / "metrics.json").resolve()),
        best_epoch=best_epoch,
        completed_epochs=len(history),
        monai=__import__("monai").__version__,
    )
    logger.info("saved best checkpoint and validation artifacts to %s", run_dir)
    return run_dir


def evaluate_validation(checkpoint_path: Path, output_dir: Path) -> Path:
    _require_new_or_empty_output_dir(output_dir)
    checkpoint = _load_compatible_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    seed = int(checkpoint["seed"])
    seed_everything(seed)
    cases = list_cases(resolve_path(config["data_root"], env_var="BRATS_ROOT"))
    training_cases, validation_cases = grouped_train_validation_split(
        cases, float(config.get("validation_fraction", 0.2)), seed
    )
    device = torch.device(runtime_device(str(config.get("device", "auto"))))
    model = _model(config, checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loss, rows, samples = _evaluate_cases(
        model, validation_cases, config, device, output_dir=output_dir / "region_masks"
    )
    _save_evaluation(
        output_dir,
        rows,
        _summarize(rows, "grouped_validation", loss, _regions(config)),
        samples,
    )
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "split": "regenerated grouped validation partition",
                "seed": seed,
                "case_count": len(validation_cases),
                "training_case_count": len(training_cases),
                "training_group_count": len(
                    {case.patient_group for case in training_cases}
                ),
                "validation_group_count": len(
                    {case.patient_group for case in validation_cases}
                ),
                "training_case_ids": sorted(case.case_id for case in training_cases),
                "validation_case_ids": sorted(case.case_id for case in validation_cases),
                "training_group_ids": sorted(
                    {case.patient_group for case in training_cases}
                ),
                "validation_group_ids": sorted(
                    {case.patient_group for case in validation_cases}
                ),
                "grouping_rule": "case folder prefix before final hyphen",
                "patient_group_identity_verified": False,
                "data_root": str(resolve_path(config["data_root"], env_var="BRATS_ROOT")),
                "official_test_split": False,
                "model_name": checkpoint["model_name"],
                "resolved_config": config,
                "device": str(device),
                "torch_version": torch.__version__,
                "monai_version": __import__("monai").__version__,
                "prediction_threshold": 0.5,
                "ontology": _ontology_metadata(),
                "region_order": list(REGION_ORDER),
                "region_definitions": _regions(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def predict_case(
    checkpoint_path: Path,
    case_directory: Path,
    output_dir: Path,
) -> Path:
    from src.data.brats import load_input_volume

    _require_new_or_empty_output_dir(output_dir)
    checkpoint = _load_compatible_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    case = BraTSCase(
        case_id=case_directory.name,
        patient_group=case_directory.name.rsplit("-", 1)[0],
        directory=case_directory,
    )
    volume = load_input_volume(
        case,
        modalities=_modalities(config),
        orientation=str(config.get("orientation", "RAS")),
    )
    device = torch.device(runtime_device(str(config.get("device", "auto"))))
    model = _model(config, checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    logits = _predict_logits(
        model,
        volume["image"],
        tuple(int(value) for value in config.get("patch_size", [96, 96, 96])),
        device,
    )
    prediction = torch.sigmoid(logits).cpu().numpy() >= 0.5
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_prediction_nifti(
        output_dir, case.case_id, prediction, list(_regions(config)), volume["affine"]
    )
    (output_dir / "prediction_metadata.json").write_text(
        json.dumps(
            {
                "pipeline": f"brats_3d_{checkpoint['model_name']}",
                "checkpoint": str(checkpoint_path.resolve()),
                "seed": int(checkpoint["seed"]),
                "case_id": case.case_id,
                "case_directory": str(case_directory.resolve()),
                "modalities": list(_modalities(config)),
                "region_definitions": _regions(config),
                "region_order": list(REGION_ORDER),
                "prediction_threshold": 0.5,
                "ontology": _ontology_metadata(),
                "orientation": config.get("orientation", "RAS"),
                "resolved_config": config,
                "device": str(device),
                "torch_version": torch.__version__,
                "monai_version": __import__("monai").__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train/evaluate MONAI BraTS 3D models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="2_3d_segmentation/configs/segmentation3d.yaml")
    train_parser.add_argument("--model", choices=(
        "segresnet", "swinunetr", "resnet3d", "efficientnet3d"
    ), required=True)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint", required=True, type=Path)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", required=True, type=Path)
    predict_parser.add_argument("--case-dir", required=True, type=Path)
    predict_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "train":
        print(train(load_config(args.config), args.model))
    elif args.command == "evaluate":
        print(evaluate_validation(args.checkpoint, args.output_dir))
    else:
        print(predict_case(args.checkpoint, args.case_dir, args.output_dir))
    return 0
