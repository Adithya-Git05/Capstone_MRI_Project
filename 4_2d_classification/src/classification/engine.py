from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.common.config import load_config, resolve_path, runtime_device
from src.common.metrics import (
    classification_metrics,
    fit_temperature,
    probability_quality_metrics,
)
from src.common.reproducibility import seed_everything, worker_seed
from src.common.run import (
    configure_logging,
    create_run_dir,
    update_run_metadata,
    write_run_metadata,
)
from src.data.brisc import (
    CLASS_NAMES,
    BRISCClassificationDataset,
    ClassificationRecord,
    classification_image_hashes,
    classification_records,
    duplicate_aware_split,
)
from src.classification.model import (
    BRISCClassifier,
    EfficientNetB0Classifier,
    ResNet50Classifier,
)


def _make_model(config: dict[str, Any]) -> nn.Module:
    model_config = config.get("model", {})
    architecture = str(model_config.get("architecture", "brisc_cnn")).lower()
    if architecture == "resnet50":
        return ResNet50Classifier(num_classes=len(CLASS_NAMES))
    if architecture == "efficientnet_b0":
        return EfficientNetB0Classifier(num_classes=len(CLASS_NAMES))
    if architecture != "brisc_cnn":
        raise ValueError(f"Unsupported classification architecture: {architecture}")
    return BRISCClassifier(
        width=int(model_config.get("width", 32)),
        dropout=float(model_config.get("dropout", 0.25)),
        num_classes=len(CLASS_NAMES),
    )


def _loader(dataset: BRISCClassificationDataset, config: dict[str, Any], seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        generator=generator,
        worker_init_fn=worker_seed,
        pin_memory=config.get("device") == "cuda",
    )


def _augmentation_config(config: dict[str, Any]) -> dict[str, float | bool]:
    augmentation = config.get("augmentation", {})
    if not isinstance(augmentation, dict):
        raise ValueError("augmentation must be a YAML mapping")
    return {
        "enabled": bool(augmentation.get("enabled", False)),
        "horizontal_flip_probability": float(
            augmentation.get("horizontal_flip_probability", 0.5)
        ),
        "rotation_degrees": float(augmentation.get("rotation_degrees", 7.0)),
        "brightness_delta": float(augmentation.get("brightness_delta", 0.08)),
        "contrast_delta": float(augmentation.get("contrast_delta", 0.08)),
    }


def _dataset(
    records: list[ClassificationRecord],
    config: dict[str, Any],
    *,
    augment: bool,
) -> BRISCClassificationDataset:
    augmentation = _augmentation_config(config)
    return BRISCClassificationDataset(
        records,
        int(config.get("image_size", 160)),
        resize_mode=str(config.get("resize_mode", "stretch")),
        augment=augment and bool(augmentation["enabled"]),
        augmentation=augmentation,
    )


def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    targets: list[int] = []
    logits_list: list[np.ndarray] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for images, labels in loader:
            labels = labels.to(device)
            logits = model(images.to(device))
            total_loss += float(criterion(logits, labels).item())
            logits_list.extend(logits.cpu().numpy())
            targets.extend(labels.cpu().numpy().tolist())
    if not targets:
        raise ValueError("Cannot evaluate an empty split")
    return np.asarray(targets), np.asarray(logits_list), total_loss / len(targets)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled_logits = torch.as_tensor(logits, dtype=torch.float64) / temperature
    return torch.softmax(scaled_logits, dim=1).numpy()


def _add_probability_metrics(
    metrics: dict[str, Any], targets: np.ndarray, probabilities: np.ndarray
) -> None:
    metrics["probability_quality"] = probability_quality_metrics(
        targets, probabilities
    )


def _save_metrics(run_dir: Path, metrics: dict[str, Any], split: str, loss: float) -> None:
    metrics["split"] = split
    output_metrics = dict(metrics)
    output_metrics["split"] = split
    output_metrics["cross_entropy_loss"] = loss
    (run_dir / "metrics.json").write_text(
        json.dumps(output_metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (run_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_metrics["per_class"][0]))
        writer.writeheader()
        writer.writerows(output_metrics["per_class"])


def _save_metric_figures(
    run_dir: Path,
    metrics: dict[str, Any],
    epoch_history: list[dict[str, float]] | None = None,
    calibrated_metrics: dict[str, Any] | None = None,
) -> None:
    names = metrics["class_names"]
    reports = [("", metrics)]
    if calibrated_metrics is not None:
        reports.append(("calibrated_", calibrated_metrics))
    for prefix, report in reports:
        confusion = np.asarray(report["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
        image = axis.imshow(confusion, cmap="Blues")
        figure.colorbar(image, ax=axis, fraction=0.046)
        axis.set(
            xticks=range(len(names)), yticks=range(len(names)),
            xticklabels=names, yticklabels=names,
        )
        axis.set(
            xlabel="Predicted", ylabel="True",
            title=(
                f"{prefix.replace('_', ' ').title()}confusion matrix "
                f"({report.get('split', metrics.get('split', 'validation'))})"
            ),
        )
        axis.tick_params(axis="x", labelrotation=30)
        midpoint = confusion.max() / 2 if confusion.size else 0
        for row in range(len(names)):
            for column in range(len(names)):
                axis.text(
                    column, row, str(confusion[row, column]), ha="center",
                    va="center",
                    color="white" if confusion[row, column] > midpoint else "black",
                )
        figure.savefig(run_dir / f"{prefix}confusion_matrix.png", dpi=300)
        figure.savefig(run_dir / f"{prefix}confusion_matrix.pdf")
        plt.close(figure)

        for metric_name, x_key, y_key, x_label, y_label in (
            ("roc", "fpr", "tpr", "False-positive rate", "True-positive rate"),
            ("precision_recall", "recall", "precision", "Recall", "Precision"),
        ):
            figure, axis = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
            valid = 0
            for name in names:
                curve = report["curves_ovr"][name]
                if not curve[x_key]:
                    continue
                valid += 1
                auc_key = "auroc" if metric_name == "roc" else "auprc"
                axis.plot(
                    curve[x_key], curve[y_key],
                    label=f"{name} (AUC={curve[auc_key]:.3f})",
                )
            if valid:
                if metric_name == "roc":
                    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
                axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=x_label, ylabel=y_label)
                axis.legend(loc="best", fontsize="small")
            else:
                axis.text(
                    0.5, 0.5, "AU curve unavailable: split lacks positive or negative samples",
                    ha="center", va="center", wrap=True,
                )
            axis.set_title(
                f"{prefix.replace('_', ' ').title()}{metric_name.replace('_', ' ').title()} "
                f"({report.get('split', metrics.get('split', 'validation'))})"
            )
            figure.savefig(run_dir / f"{prefix}{metric_name}.png", dpi=300)
            figure.savefig(run_dir / f"{prefix}{metric_name}.pdf")
            plt.close(figure)

    quality_reports = [("Uncalibrated", metrics["probability_quality"])]
    if calibrated_metrics is not None:
        quality_reports.append(("Temperature-scaled", calibrated_metrics["probability_quality"]))
    figure, axis = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    for label, quality in quality_reports:
        bins = quality["reliability_bins"]
        populated = [item for item in bins if item["count"] > 0]
        axis.plot(
            [item["mean_confidence"] for item in populated],
            [item["accuracy"] for item in populated],
            marker="o", label=label,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set(
        xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence",
        ylabel="Accuracy", title=f"Reliability diagram ({metrics['split']})",
    )
    axis.legend(loc="best")
    figure.savefig(run_dir / "reliability.png", dpi=300)
    figure.savefig(run_dir / "reliability.pdf")
    plt.close(figure)

    if epoch_history:
        epochs = [entry["epoch"] for entry in epoch_history]
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        axes[0].plot(epochs, [entry["train_loss"] for entry in epoch_history], label="train")
        axes[0].plot(epochs, [entry["validation_loss"] for entry in epoch_history], label="validation")
        axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
        axes[0].legend()
        axes[1].plot(epochs, [entry["validation_macro_f1"] for entry in epoch_history], label="macro F1")
        axes[1].plot(epochs, [entry["validation_balanced_accuracy"] for entry in epoch_history], label="balanced accuracy")
        axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics", ylim=(0, 1))
        axes[1].legend()
        axes[2].plot(epochs, [entry["learning_rate"] for entry in epoch_history])
        axes[2].set(xlabel="Epoch", ylabel="Learning rate", title="Learning-rate schedule")
        figure.savefig(run_dir / "training_curves.png", dpi=300)
        figure.savefig(run_dir / "training_curves.pdf")
        plt.close(figure)


def train(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed", 2026))
    seed_everything(seed)
    data_root = resolve_path(config["data_root"], env_var="BRISC_ROOT")
    output_root = resolve_path(config["output_root"], env_var="OUTPUT_ROOT")
    official_train_records = classification_records(data_root, "train")
    test_records = classification_records(data_root, "test")
    official_train_hashes = classification_image_hashes(official_train_records)
    test_hashes = classification_image_hashes(test_records)
    test_hash_set = set(test_hashes)
    train_test_overlap = sorted(set(official_train_hashes) & test_hash_set)
    excluded_train_records = [
        record
        for record, digest in zip(
            official_train_records, official_train_hashes, strict=True
        )
        if digest in test_hash_set
    ]
    records = [
        record
        for record, digest in zip(
            official_train_records, official_train_hashes, strict=True
        )
        if digest not in test_hash_set
    ]
    if not records:
        raise ValueError("No official training images remain after excluding test duplicates")
    validation_fraction = float(config.get("validation_fraction", 0.15))
    calibration_fraction = float(config.get("calibration_fraction", 0.1))
    train_indices, validation_indices, calibration_indices = duplicate_aware_split(
        records, validation_fraction, calibration_fraction, seed
    )
    split_by_hash: dict[str, str] = {}
    split_records = {
        "model_fit": [records[index] for index in train_indices],
        "validation": [records[index] for index in validation_indices],
        "calibration": [records[index] for index in calibration_indices],
        "excluded_train_test_duplicates": excluded_train_records,
    }
    for partition, partition_records in split_records.items():
        if partition == "excluded_train_test_duplicates":
            continue
        for digest in classification_image_hashes(partition_records):
            existing_partition = split_by_hash.get(digest)
            if existing_partition is not None and existing_partition != partition:
                raise RuntimeError("Identical image hashes crossed internal partitions")
            split_by_hash[digest] = partition
    image_size = int(config.get("image_size", 160))
    preprocessing_config = {
    "resize_mode": str(config.get("resize_mode", "stretch")),
        "augmentation": _augmentation_config(config),
    }
    train_loader = _loader(
        _dataset(split_records["model_fit"], config, augment=True),
        config, seed, True,
    )
    validation_loader = _loader(
        _dataset(split_records["validation"], config, augment=False),
        config, seed, False,
    )
    calibration_loader = _loader(
        _dataset(split_records["calibration"], config, augment=False),
        config, seed, False,
    )
    device = torch.device(runtime_device(str(config.get("device", "auto"))))
    model = _make_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )
    scheduler_config = config.get("scheduler", {})
    if not isinstance(scheduler_config, dict):
        raise ValueError("scheduler must be a YAML mapping")
    scheduler_enabled = bool(scheduler_config.get("enabled", False))
    scheduler_name = str(scheduler_config.get("name", "cosine")).lower()
    if scheduler_enabled and scheduler_name != "cosine":
        raise ValueError("Only the cosine learning-rate scheduler is supported")
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("epochs", 30)),
            eta_min=float(scheduler_config.get("eta_min", 1e-6)),
        )
        if scheduler_enabled
        else None
    )
    run_dir = create_run_dir(output_root, f"seed{seed}")
    logger = configure_logging(run_dir)
    manifest_checksums = {
        name: _sha256_file(data_root / name)
        for name in ("manifest.csv", "manifest.json")
    }
    class_counts = {
        partition: {
            class_name: sum(record.label == label for record in partition_records)
            for label, class_name in enumerate(CLASS_NAMES)
        }
        for partition, partition_records in split_records.items()
    }
    resolved_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    resolved_config.update(
        data_root=str(data_root.resolve()),
        output_root=str(output_root.resolve()),
        device=str(device),
    )
    if config.get("_config_path") is not None:
        resolved_config["_config_path"] = config["_config_path"]
    write_run_metadata(
        run_dir, resolved_config, pipeline="brisc_2d_classification",
        split_details={
            "training_count": len(train_indices),
            "validation_count": len(validation_indices),
            "calibration_count": len(calibration_indices),
            "official_training_count": len(official_train_records),
            "official_training_excluded_test_duplicate_count": len(excluded_train_records),
            "eligible_training_count": len(records),
            "partition_counts_by_class": class_counts,
            "training_split": (
                "official train only; exact test-overlap copies excluded; "
                "remaining duplicate hashes grouped"
            ),
            "test_split_used_for_selection": False,
            "grouping": "exact image SHA-256; patient identifiers unavailable",
            "internal_hash_partitions_disjoint": True,
            "official_test_membership_preserved": True,
            "official_train_test_hash_overlap_count": len(train_test_overlap),
            "official_train_test_hash_overlap": train_test_overlap,
            "official_train_test_overlapping_train_file_count": len(excluded_train_records),
            "official_test_count": len(test_records),
        },
    )
    update_run_metadata(
        run_dir,
        device=str(device),
        data_root=str(data_root.resolve()),
        manifest_sha256=manifest_checksums,
        checkpoint_selection={
            "criterion": "minimum validation cross-entropy",
            "early_stopping_patience": int(config.get("early_stopping_patience", 7)),
            "validation_used_for_checkpoint_selection": True,
            "calibration_used_for_checkpoint_selection": False,
            "official_test_used_for_checkpoint_selection": False,
        },
        scheduler={
            "enabled": scheduler_enabled,
            "name": scheduler_name if scheduler_enabled else None,
            "eta_min": float(scheduler_config.get("eta_min", 1e-6)) if scheduler_enabled else None,
        },
        preprocessing=preprocessing_config,
        class_names=list(CLASS_NAMES),
    )
    with (run_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("path", "sha256", "label", "partition")
        )
        writer.writeheader()
        for partition, partition_records in split_records.items():
            for record, digest in zip(
                partition_records, classification_image_hashes(partition_records), strict=True
            ):
                writer.writerow(
                    {
                        "path": str(record.path.relative_to(data_root)),
                        "sha256": digest,
                        "label": CLASS_NAMES[record.label],
                        "partition": partition,
                    }
                )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    patience = int(config.get("early_stopping_patience", 7))
    epochs = int(config.get("epochs", 30))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        validation_targets, validation_logits, validation_loss = _predict(
            model, validation_loader, device
        )
        validation_probabilities = _probabilities(validation_logits, 1.0)
        validation_metrics = classification_metrics(
            validation_targets, validation_probabilities, list(CLASS_NAMES)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "learning_rate": learning_rate,
            }
        )
        logger.info(
            "epoch=%d train_loss=%.5f validation_loss=%.5f validation_accuracy=%.4f lr=%.7g",
            epoch, train_loss, validation_loss, validation_metrics["accuracy"], learning_rate,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config.get("model", {}),
                    "class_names": list(CLASS_NAMES),
                    "image_size": image_size,
                    "preprocessing": preprocessing_config,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "temperature": 1.0,
                    "resolved_config": {
                        key: value for key, value in resolved_config.items()
                        if not key.startswith("_")
                    },
                    "manifest_sha256": manifest_checksums,
                },
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("early stopping at epoch %d; best epoch=%d", epoch, best_epoch)
                break
        if scheduler is not None:
            scheduler.step()

    history_path = run_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    calibration_targets, calibration_logits, _ = _predict(
        model, calibration_loader, device
    )
    temperature = fit_temperature(calibration_logits, calibration_targets)
    calibration_metadata = {
        "method": "scalar temperature scaling",
        "temperature": temperature,
        "sample_count": len(calibration_indices),
        "partition": "official train calibration subset",
        "used_for_model_fitting": False,
        "used_for_checkpoint_selection": False,
        "fit_objective": "calibration-subset cross-entropy",
    }
    checkpoint["temperature"] = temperature
    checkpoint["calibration"] = calibration_metadata
    torch.save(checkpoint, run_dir / "best.pt")
    validation_targets, validation_logits, validation_loss = _predict(
        model, validation_loader, device
    )
    uncalibrated_probabilities = _probabilities(validation_logits, 1.0)
    calibrated_probabilities = _probabilities(validation_logits, temperature)
    metrics = classification_metrics(
        validation_targets, uncalibrated_probabilities, list(CLASS_NAMES)
    )
    calibrated_metrics = classification_metrics(
        validation_targets, calibrated_probabilities, list(CLASS_NAMES)
    )
    _add_probability_metrics(metrics, validation_targets, uncalibrated_probabilities)
    _add_probability_metrics(calibrated_metrics, validation_targets, calibrated_probabilities)
    calibrated_metrics["split"] = "validation"
    _save_metrics(run_dir, metrics, "validation", validation_loss)
    (run_dir / "calibrated_metrics.json").write_text(
        json.dumps(
            {**calibrated_metrics, "split": "validation", "cross_entropy_loss": validation_loss},
            indent=2, allow_nan=False,
        ),
        encoding="utf-8",
    )
    with (run_dir / "calibrated_per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(calibrated_metrics["per_class"][0])
        )
        writer.writeheader()
        writer.writerows(calibrated_metrics["per_class"])
    _save_metric_figures(run_dir, metrics, history, calibrated_metrics)
    update_run_metadata(
        run_dir,
        checkpoint=str((run_dir / "best.pt").resolve()),
        metrics=str((run_dir / "metrics.json").resolve()),
        calibrated_metrics=str((run_dir / "calibrated_metrics.json").resolve()),
        best_epoch=best_epoch,
        completed_epochs=len(history),
        sample_count=len(official_train_records),
        training_count=len(train_indices),
        validation_count=len(validation_indices),
        calibration_count=len(calibration_indices),
        eligible_training_count=len(records),
        excluded_train_test_duplicate_count=len(excluded_train_records),
        calibration=calibration_metadata,
        official_test_hash_overlap_report={
            "overlap_count": len(train_test_overlap),
            "sha256": train_test_overlap,
            "test_membership_modified": False,
            "limitation": "Exact duplicate pixels occur across the official train and test splits.",
        },
    )
    logger.info("saved best checkpoint and validation artifacts to %s", run_dir)
    return run_dir


def evaluate(
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path | None,
    split: str,
) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    class_names = checkpoint.get("class_names")
    if class_names != list(CLASS_NAMES):
        raise ValueError(f"Checkpoint class order does not match BRISC: {class_names}")
    config = {
        "model": checkpoint["model_config"],
        "image_size": checkpoint["image_size"],
        "resize_mode": checkpoint.get("preprocessing", {}).get("resize_mode", "stretch"),
        "augmentation": checkpoint.get("preprocessing", {}).get("augmentation", {}),
        "device": "auto",
        "batch_size": 32,
        "num_workers": 0,
    }
    device = torch.device(runtime_device())
    model = _make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    records = classification_records(data_root, split)
    loader = _loader(_dataset(records, config, augment=False), config, 0, False)
    targets, logits, loss = _predict(model, loader, device)
    temperature = float(checkpoint.get("temperature", 1.0))
    uncalibrated_probabilities = _probabilities(logits, 1.0)
    calibrated_probabilities = _probabilities(logits, temperature)
    metrics = classification_metrics(
        targets, uncalibrated_probabilities, list(CLASS_NAMES)
    )
    calibrated_metrics = classification_metrics(
        targets, calibrated_probabilities, list(CLASS_NAMES)
    )
    _add_probability_metrics(metrics, targets, uncalibrated_probabilities)
    _add_probability_metrics(calibrated_metrics, targets, calibrated_probabilities)
    if output_dir is None:
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = checkpoint_path.parent / f"evaluation_{split}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    _save_metrics(output_dir, metrics, split, loss)
    (output_dir / "calibrated_metrics.json").write_text(
        json.dumps(
            {**calibrated_metrics, "split": split, "cross_entropy_loss": loss},
            indent=2, allow_nan=False,
        ),
        encoding="utf-8",
    )
    with (output_dir / "calibrated_per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(calibrated_metrics["per_class"][0])
        )
        writer.writeheader()
        writer.writerows(calibrated_metrics["per_class"])
    _save_metric_figures(output_dir, metrics, calibrated_metrics=calibrated_metrics)
    official_train_hashes = classification_image_hashes(
        classification_records(data_root, "train")
    )
    evaluated_hashes = classification_image_hashes(records)
    overlap_hashes = sorted(set(official_train_hashes) & set(evaluated_hashes))
    current_manifest_checksums = {
        name: _sha256_file(data_root / name)
        for name in ("manifest.csv", "manifest.json")
    }
    metadata = {
        "pipeline": "brisc_2d_classification_evaluation",
        "checkpoint": str(checkpoint_path),
        "split": split,
        "sample_count": len(records),
        "data_root": str(data_root.resolve()),
        "test_is_official": split == "test",
        "official_split_membership_modified": False,
        "official_train_evaluation_hash_overlap_count": len(overlap_hashes),
        "official_train_evaluation_hash_overlap": overlap_hashes,
        "dataset_limitation": (
            "Exact duplicate images occur across official train and test splits."
            if split == "test" and overlap_hashes
            else None
        ),
        "temperature": temperature,
        "calibration": checkpoint.get("calibration", {
            "method": "none; legacy checkpoint fallback",
            "temperature": temperature,
        }),
        "manifest_sha256": current_manifest_checksums,
        "checkpoint_manifest_sha256": checkpoint.get("manifest_sha256"),
        "resolved_config": checkpoint.get("resolved_config"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    if checkpoint.get("resolved_config") is not None:
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump(checkpoint["resolved_config"], sort_keys=True),
            encoding="utf-8",
        )
    return output_dir


def predict(checkpoint_path: Path, input_path: Path, output_path: Path) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Checkpoint class order does not match BRISC")
    device = torch.device(runtime_device())
    model_config = {
        "model": checkpoint["model_config"],
        "image_size": checkpoint["image_size"],
        "resize_mode": checkpoint.get("preprocessing", {}).get("resize_mode", "stretch"),
        "augmentation": checkpoint.get("preprocessing", {}).get("augmentation", {}),
        "device": "auto",
    }
    model = _make_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    temperature = float(checkpoint.get("temperature", 1.0))
    paths = sorted(input_path.iterdir()) if input_path.is_dir() else [input_path]
    rows = []
    with torch.inference_mode():
        for image_path in paths:
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image, _ = _dataset(
                [ClassificationRecord(image_path, 0)], model_config, augment=False
            )[0]
            probability = torch.softmax(
                model(image.unsqueeze(0).to(device)) / temperature, dim=1
            )[0].cpu().numpy()
            predicted = int(probability.argmax())
            rows.append(
                {
                    "image": str(image_path),
                    "predicted_class": CLASS_NAMES[predicted],
                    "confidence": float(probability[predicted]),
                    **{f"probability_{name}": float(probability[i]) for i, name in enumerate(CLASS_NAMES)},
                }
            )
    if not rows:
        raise ValueError(f"No supported image files found at {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata_path = output_path.with_name("prediction_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "pipeline": "brisc_2d_classification",
                "checkpoint": str(checkpoint_path.resolve()),
                "input": str(input_path.resolve()),
                "data_root": str(input_path.resolve()) if input_path.is_dir() else str(input_path.parent.resolve()),
                "sample_count": len(rows),
                "class_names": list(CLASS_NAMES),
                "image_size": checkpoint["image_size"],
                "resize_mode": model_config["resize_mode"],
                "temperature": temperature,
                "calibration": checkpoint.get("calibration", {
                    "method": "none; legacy checkpoint fallback",
                    "temperature": temperature,
                }),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="BRISC 2D classification")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="4_2d_classification/configs/classification.yaml")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument("--data-root", type=Path)
    evaluate_parser.add_argument("--output-dir", type=Path)
    evaluate_parser.add_argument("--split", choices=("train", "test"), default="test")
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", required=True, type=Path)
    predict_parser.add_argument("--input", required=True, type=Path)
    predict_parser.add_argument("--output", type=Path, default=Path("4_2d_classification/outputs/classification/predictions.csv"))
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        print(train(config))
    elif args.command == "evaluate":
        data_root = args.data_root or resolve_path("1_data_and_preprocessing/data/BRISC/brisc2025", env_var="BRISC_ROOT")
        output_dir = args.output_dir
        print(evaluate(args.checkpoint, data_root, output_dir, args.split))
    else:
        print(predict(args.checkpoint, args.input, args.output))
    return 0
