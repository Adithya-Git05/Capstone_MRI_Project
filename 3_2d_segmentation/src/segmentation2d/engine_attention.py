from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.common.config import load_config, resolve_path, runtime_device
from src.common.metrics import binary_segmentation_metrics
from src.common.reproducibility import seed_everything, worker_seed
from src.data.brisc import BRISCSegmentationDataset, SegmentationRecord, segmentation_records
from src.segmentation2d.model_attention import BRISCAttentionUNet


def _model(config: dict[str, Any]) -> BRISCAttentionUNet:
    return BRISCAttentionUNet(
        base_channels=int(config.get("model", {}).get("base_channels", 16))
    )


def _loader(
    records: list[SegmentationRecord],
    config: dict[str, Any],
    seed: int,
    *,
    shuffle: bool,
    augment: bool = False,
) -> DataLoader:
    dataset = BRISCSegmentationDataset(
        records,
        int(config.get("image_size", 256)),
        int(config.get("mask_threshold", 128)),
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=worker_seed,
        pin_memory=config.get("device") == "cuda",
    )


def _loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float = 0.10,
) -> torch.Tensor:
    if not 0 <= boundary_weight <= 1:
        raise ValueError("boundary_weight must be in [0, 1]")
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

    pred_dx = probabilities[..., 1:] - probabilities[..., :-1]
    true_dx = target[..., 1:] - target[..., :-1]
    pred_dy = probabilities[..., 1:, :] - probabilities[..., :-1, :]
    true_dy = target[..., 1:, :] - target[..., :-1, :]
    boundary_loss = (
        nn.functional.l1_loss(pred_dx, true_dx)
        + nn.functional.l1_loss(pred_dy, true_dy)
    ) / 2.0
    base_loss = 0.5 * bce + 0.5 * dice_loss
    return (1.0 - boundary_weight) * base_loss + boundary_weight * boundary_loss


def _evaluate_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    boundary_weight: float = 0.10,
) -> tuple[
    float,
    list[dict[str, Any]],
    list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
]:
    model.eval()
    rows: list[dict[str, Any]] = []
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    total_loss = 0.0
    offset = 0
    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            targets = masks.to(device)
            total_loss += float(_loss(logits, targets, boundary_weight).item()) * len(images)
            predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy()[:, 0]
            truths = masks.cpu().numpy()[:, 0] >= 0.5
            for batch_index, prediction in enumerate(predictions):
                truth = truths[batch_index]
                scores = binary_segmentation_metrics(prediction, truth)
                record = loader.dataset.records[offset + batch_index]
                rows.append(
                    {
                        "sample_index": offset + batch_index,
                        "sample_id": record.image_path.stem,
                        "image_path": str(record.image_path),
                        "mask_path": str(record.mask_path),
                        "dice": float(scores["dice"]),
                        "iou": float(scores["iou"]),
                        "target_foreground_pixels": int(truth.sum()),
                        "predicted_foreground_pixels": int(prediction.sum()),
                        "intersection_pixels": int(
                            np.logical_and(prediction, truth).sum()
                        ),
                    }
                )
                if len(samples) < 8:
                    samples.append(
                        (
                            images[batch_index, 0].cpu().numpy(),
                            truth,
                            prediction,
                            record.image_path.stem,
                        )
                    )
            offset += len(images)
    if not rows:
        raise ValueError("Cannot evaluate an empty segmentation split")
    return total_loss / len(rows), rows, samples


def _aggregate(
    rows: list[dict[str, Any]], split: str, loss: float
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate metrics for an empty split")
    intersection = sum(row["intersection_pixels"] for row in rows)
    predicted = sum(row["predicted_foreground_pixels"] for row in rows)
    target = sum(row["target_foreground_pixels"] for row in rows)
    denominator = predicted + target
    union = predicted + target - intersection
    nonempty_rows = [row for row in rows if row["target_foreground_pixels"] > 0]
    return {
        "split": split,
        "sample_count": len(rows),
        "boundary_bce_soft_dice_loss": loss,
        "global_foreground_dice": float(2 * intersection / denominator) if denominator else 1.0,
        "global_foreground_iou": float(intersection / union) if union else 1.0,
        "mean_per_image_dice_including_empty_targets": float(
            np.mean([row["dice"] for row in rows])
        ),
        "mean_per_image_iou_including_empty_targets": float(
            np.mean([row["iou"] for row in rows])
        ),
        "mean_per_image_dice_nonempty_targets": (
            float(np.mean([row["dice"] for row in nonempty_rows])
        ) if nonempty_rows else None),
        "nonempty_target_count": len(nonempty_rows),
        "empty_target_count": len(rows) - len(nonempty_rows),
    }


def _save_evaluation_artifacts(
    output_dir: Path,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
) -> None:
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].hist([row["dice"] for row in rows], bins=20, range=(0, 1))
    axes[0].set(xlabel="Per-image Dice", ylabel="Image count", title="Test Dice distribution")
    axes[1].hist([row["iou"] for row in rows], bins=20, range=(0, 1))
    axes[1].set(xlabel="Per-image IoU", ylabel="Image count", title="Test IoU distribution")
    figure.savefig(output_dir / "test_metric_distributions.png", dpi=300)
    figure.savefig(output_dir / "test_metric_distributions.pdf")
    plt.close(figure)

    if samples:
        figure, axes = plt.subplots(
            len(samples), 3, figsize=(9, 3 * len(samples)), squeeze=False
        )
        for index, (image, target, prediction, sample_id) in enumerate(samples):
            display = (image * 0.5 + 0.5).clip(0, 1)
            axes[index, 0].imshow(display, cmap="gray")
            axes[index, 0].set_title(f"{sample_id} MRI")
            axes[index, 1].imshow(display, cmap="gray")
            axes[index, 1].imshow(target, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
            axes[index, 1].set_title("Ground truth overlay")
            axes[index, 2].imshow(display, cmap="gray")
            axes[index, 2].imshow(prediction, cmap="Blues", alpha=0.45, vmin=0, vmax=1)
            axes[index, 2].set_title("Prediction overlay")
            for axis in axes[index]:
                axis.axis("off")
        figure.savefig(output_dir / "representative_test_overlays.png", dpi=300)
        figure.savefig(output_dir / "representative_test_overlays.pdf")
        plt.close(figure)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_config(checkpoint_path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    if config.get("model", {}).get("base_channels") != checkpoint["model_config"].get(
        "base_channels"
    ):
        raise ValueError("Experiment config base_channels does not match checkpoint")
    if int(config.get("image_size", -1)) != int(checkpoint["image_size"]):
        raise ValueError("Experiment config image_size does not match checkpoint")
    if int(config.get("mask_threshold", -1)) != int(checkpoint["mask_threshold"]):
        raise ValueError("Experiment config mask_threshold does not match checkpoint")
    return config


def _create_evaluation_dir(
    checkpoint_path: Path, split: str, output_dir: Path | None
) -> Path:
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = checkpoint_path.parent / f"evaluation_{split}_{timestamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def evaluate(
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path | None,
    split: str,
    *,
    archive_source: Path | None = None,
) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"model_state", "model_config", "image_size", "mask_threshold"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
    config = _experiment_config(checkpoint_path, checkpoint)
    model = _model(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    output_dir = _create_evaluation_dir(checkpoint_path, split, output_dir)

    records = segmentation_records(data_root, split)
    if split == "test" and len(records) != 860:
        raise ValueError(
            f"Expected 860 official BRISC segmentation test pairs, found {len(records)}"
        )
    for record in records:
        if not record.image_path.is_file() or not record.mask_path.is_file():
            raise FileNotFoundError(
                f"Missing paired BRISC file: {record.image_path} / {record.mask_path}"
            )

    evaluation_config = dict(config)
    evaluation_config.update(device="auto", num_workers=0)
    device = torch.device(runtime_device())
    model = model.to(device)
    loader = _loader(records, evaluation_config, int(checkpoint.get("seed", 0)), shuffle=False)
    loss, rows, samples = _evaluate_with_predictions(
        model,
        loader,
        device,
        boundary_weight=float(config.get("boundary_weight", 0.10)),
    )
    metrics = _aggregate(rows, split, loss)

    _save_evaluation_artifacts(output_dir, metrics, rows, samples)

    if archive_source is not None and not archive_source.is_file():
        raise FileNotFoundError(f"Experiment archive not found: {archive_source}")
    metadata = {
        "pipeline": "brisc_2d_attention_boundary_segmentation_evaluation",
        "model": "BRISCAttentionUNet",
        "model_config": checkpoint["model_config"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "experiment_config": str((checkpoint_path.parent / "config.yaml").resolve()),
        "resolved_experiment_config": config,
        "experiment_archive": str(archive_source.resolve()) if archive_source else None,
        "experiment_archive_sha256": _sha256_file(archive_source) if archive_source else None,
        "split": split,
        "official_test_split": split == "test",
        "official_test_pair_count": len(records),
        "pairing": "image/mask stems validated by segmentation_records",
        "preprocessing": {
            "image": "grayscale, bilinear resize to checkpoint image_size, normalize to [-1, 1]",
            "mask": (
                f"grayscale threshold >= {checkpoint['mask_threshold']}, "
                "nearest-neighbor resize to checkpoint image_size"
            ),
            "threshold_prediction": 0.5,
        },
        "image_size": int(checkpoint["image_size"]),
        "mask_threshold": int(checkpoint["mask_threshold"]),
        "boundary_loss_weight_from_training_code": float(config.get("boundary_weight", 0.10)),
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    return output_dir


def train(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed", 2026))
    seed_everything(seed)
    data_root = resolve_path(config["data_root"], env_var="BRISC_ROOT")
    output_root = resolve_path(config["output_root"], env_var="OUTPUT_ROOT")
    records = segmentation_records(data_root, "train")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    validation_count = min(
        len(records) - 1,
        max(1, round(len(records) * float(config.get("validation_fraction", 0.15)))),
    )
    validation_indices = sorted(order[:validation_count].tolist())
    training_indices = sorted(order[validation_count:].tolist())
    training_records = [records[index] for index in training_indices]
    validation_records = [records[index] for index in validation_indices]
    training_loader = _loader(training_records, config, seed, shuffle=True, augment=True)
    validation_loader = _loader(validation_records, config, seed, shuffle=False)
    device = torch.device(runtime_device(str(config.get("device", "auto"))))
    model = _model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.0003)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )
    boundary_weight = float(config.get("boundary_weight", 0.10))
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"attention_boundary_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    best_dice = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config.get("epochs", 40)) + 1):
        model.train()
        loss_sum = 0.0
        for images, masks in training_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = _loss(model(images), masks, boundary_weight)
            batch_loss.backward()
            optimizer.step()
            loss_sum += float(batch_loss.item()) * len(images)
        train_loss = loss_sum / len(training_loader.dataset)
        validation_loss, rows, _ = _evaluate_with_predictions(
            model, validation_loader, device, boundary_weight=boundary_weight
        )
        validation_dice = float(np.mean([row["dice"] for row in rows]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_dice": validation_dice,
            }
        )
        if validation_dice > best_dice:
            best_dice, best_epoch, stale_epochs = validation_dice, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config.get("model", {}),
                    "image_size": int(config.get("image_size", 256)),
                    "mask_threshold": int(config.get("mask_threshold", 128)),
                    "seed": seed,
                    "best_epoch": best_epoch,
                },
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.get("early_stopping_patience", 8)):
                break
    with (run_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="BRISC 2D attention-boundary segmentation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="3_2d_segmentation/configs/segmentation2d.yaml")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument("--data-root", type=Path)
    evaluate_parser.add_argument("--output-dir", type=Path)
    evaluate_parser.add_argument("--archive-source", type=Path)
    evaluate_parser.add_argument("--split", choices=("train", "test"), default="test")
    args = parser.parse_args()
    if args.command == "train":
        print(train(load_config(args.config)))
    else:
        data_root = args.data_root or resolve_path(
            "1_data_and_preprocessing/data/BRISC/brisc2025", env_var="BRISC_ROOT"
        )
        archive_source = args.archive_source
        default_archive = Path(
            r"C:\Personal\BMS\Capstone\Outputs-BRISC-Segmentation\BRISC_Segmentation_Experiments.zip"
        )
        if archive_source is None and default_archive.is_file():
            archive_source = default_archive
        print(
            evaluate(
                args.checkpoint,
                data_root,
                args.output_dir,
                args.split,
                archive_source=archive_source,
            )
        )
    return 0
