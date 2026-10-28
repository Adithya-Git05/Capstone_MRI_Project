from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common.config import load_config, resolve_path, runtime_device
from src.common.metrics import binary_segmentation_metrics
from src.common.reproducibility import seed_everything, worker_seed
from src.common.run import (
    configure_logging,
    create_run_dir,
    update_run_metadata,
    write_run_metadata,
)
from src.data.brisc import (
    BRISCClassificationDataset,
    BRISCSegmentationDataset,
    ClassificationRecord,
    SegmentationRecord,
    segmentation_records,
)
from src.segmentation2d.model import (
    BRISCUNet,
    EfficientNetB0UNet,
    ResNet50UNet,
)


def _model(config: dict[str, Any]) -> nn.Module:
    architecture = str(config.get("model", {}).get("architecture", "brisc_unet")).lower()
    if architecture == "resnet50":
        return ResNet50UNet()
    if architecture == "efficientnet_b0":
        return EfficientNetB0UNet()
    if architecture != "brisc_unet":
        raise ValueError(f"Unsupported segmentation architecture: {architecture}")
    return BRISCUNet(base_channels=int(config.get("model", {}).get("base_channels", 16)))


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


def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    binary_cross_entropy = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return 0.5 * binary_cross_entropy + 0.5 * dice_loss


def _evaluate_with_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    total_loss = 0.0
    offset = 0
    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            targets = masks.to(device)
            total_loss += float(_loss(logits, targets).item()) * images.size(0)
            predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy()[:, 0]
            truths = masks.cpu().numpy()[:, 0] >= 0.5
            for batch_index, prediction in enumerate(predictions):
                truth = truths[batch_index]
                intersection = int(np.logical_and(prediction, truth).sum())
                predicted_count = int(prediction.sum())
                target_count = int(truth.sum())
                scores = binary_segmentation_metrics(prediction, truth)
                rows.append(
                    {
                        "sample_index": offset + batch_index,
                        "sample_id": loader.dataset.records[offset + batch_index].image_path.stem,
                        "dice": float(scores["dice"]),
                        "iou": float(scores["iou"]),
                        "target_foreground_pixels": target_count,
                        "predicted_foreground_pixels": predicted_count,
                        "intersection_pixels": intersection,
                    }
                )
                if len(samples) < 8:
                    image = images[batch_index, 0].cpu().numpy()
                    sample_id = loader.dataset.records[offset + batch_index].image_path.stem
                    samples.append((image, truth, prediction, sample_id))
            offset += images.size(0)
    if not rows:
        raise ValueError("Cannot evaluate an empty segmentation split")
    return total_loss / len(rows), rows, samples


def _save_segmentation_artifacts(
    run_dir: Path,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    history: list[dict[str, float]] | None = None,
) -> None:
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (run_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if history:
        with (run_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        epochs = [item["epoch"] for item in history]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
        axes[0].plot(epochs, [item["validation_loss"] for item in history], label="validation")
        axes[0].set(xlabel="Epoch", ylabel="BCE + soft Dice loss", title="Loss")
        axes[0].legend()
        axes[1].plot(epochs, [item["validation_dice"] for item in history], label="validation")
        axes[1].set(xlabel="Epoch", ylabel="Foreground Dice", ylim=(0, 1), title="Validation Dice")
        axes[1].legend()
        figure.savefig(run_dir / "training_curves.png", dpi=300)
        figure.savefig(run_dir / "training_curves.pdf")
        plt.close(figure)
    if samples:
        figure, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)), squeeze=False)
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
        figure.savefig(run_dir / "representative_overlays.png", dpi=300)
        figure.savefig(run_dir / "representative_overlays.pdf")
        plt.close(figure)


def _aggregate(rows: list[dict[str, Any]], split: str, loss: float) -> dict[str, Any]:
    intersection = sum(row["intersection_pixels"] for row in rows)
    predicted = sum(row["predicted_foreground_pixels"] for row in rows)
    target = sum(row["target_foreground_pixels"] for row in rows)
    dice_denominator = predicted + target
    union = predicted + target - intersection
    nonempty_rows = [row for row in rows if row["target_foreground_pixels"] > 0]
    return {
        "split": split,
        "sample_count": len(rows),
        "bce_plus_soft_dice_loss": loss,
        "global_foreground_dice": (
            float(2 * intersection / dice_denominator) if dice_denominator else 1.0
        ),
        "global_foreground_iou": float(intersection / union) if union else 1.0,
        "mean_per_image_dice_including_empty_targets": float(
            np.mean([row["dice"] for row in rows])
        ),
        "mean_per_image_iou_including_empty_targets": float(
            np.mean([row["iou"] for row in rows])
        ),
        "mean_per_image_dice_nonempty_targets": (
            float(np.mean([row["dice"] for row in nonempty_rows])) if nonempty_rows else None
        ),
        "nonempty_target_count": len(nonempty_rows),
        "empty_target_count": len(rows) - len(nonempty_rows),
    }


def train(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed", 2026))
    seed_everything(seed)
    data_root = resolve_path(config["data_root"], env_var="BRISC_ROOT")
    output_root = resolve_path(config["output_root"], env_var="OUTPUT_ROOT")
    records = segmentation_records(data_root, "train")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n_validation = min(len(records) - 1, max(1, round(len(records) * float(config.get("validation_fraction", 0.15)))))
    validation_indices = sorted(order[:n_validation].tolist())
    train_indices = sorted(order[n_validation:].tolist())
    train_records = [records[index] for index in train_indices]
    validation_records = [records[index] for index in validation_indices]
    train_loader = _loader(train_records, config, seed, shuffle=True, augment=True)
    validation_loader = _loader(validation_records, config, seed, shuffle=False)
    device = torch.device(runtime_device(str(config.get("device", "auto"))))
    model = _model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.0003)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )
    run_dir = create_run_dir(output_root, f"seed{seed}")
    logger = configure_logging(run_dir)
    write_run_metadata(
        run_dir,
        config,
        pipeline="brisc_2d_segmentation",
        split_details={
            "training_count": len(train_records),
            "validation_count": len(validation_records),
            "training_split": "official train subset only",
            "test_split_used_for_selection": False,
            "grouping": "sample-level; BRISC patient identifiers are unavailable",
            "mask_threshold": int(config.get("mask_threshold", 128)),
        },
    )
    update_run_metadata(run_dir, device=str(device), data_root=str(data_root.resolve()))
    history: list[dict[str, float]] = []
    best_dice = -1.0
    best_epoch = 0
    stale_epochs = 0
    patience = int(config.get("early_stopping_patience", 8))
    for epoch in range(1, int(config.get("epochs", 40)) + 1):
        model.train()
        loss_sum = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(images), masks)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * images.size(0)
        train_loss = loss_sum / len(train_loader.dataset)
        validation_loss, rows, _ = _evaluate_with_predictions(model, validation_loader, device)
        validation_dice = float(np.mean([row["dice"] for row in rows]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_dice": validation_dice,
            }
        )
        logger.info(
            "epoch=%d train_loss=%.5f validation_loss=%.5f validation_dice=%.4f",
            epoch, train_loss, validation_loss, validation_dice,
        )
        if validation_dice > best_dice:
            best_dice = validation_dice
            best_epoch = epoch
            stale_epochs = 0
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
            if stale_epochs >= patience:
                logger.info("early stopping at epoch %d; best epoch=%d", epoch, best_epoch)
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    validation_loss, rows, samples = _evaluate_with_predictions(model, validation_loader, device)
    metrics = _aggregate(rows, "validation", validation_loss)
    _save_segmentation_artifacts(run_dir, metrics, rows, samples, history)
    update_run_metadata(
        run_dir,
        checkpoint=str((run_dir / "best.pt").resolve()),
        metrics=str((run_dir / "metrics.json").resolve()),
        best_epoch=best_epoch,
        completed_epochs=len(history),
        sample_count=len(records),
        training_count=len(train_records),
        validation_count=len(validation_records),
    )
    logger.info("saved best checkpoint and validation artifacts to %s", run_dir)
    return run_dir


def evaluate(checkpoint_path: Path, data_root: Path, output_dir: Path, split: str) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = {
        "model": checkpoint["model_config"],
        "image_size": checkpoint["image_size"],
        "mask_threshold": checkpoint["mask_threshold"],
        "batch_size": 8,
        "num_workers": 0,
        "device": "auto",
    }
    device = torch.device(runtime_device())
    model = _model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    records = segmentation_records(data_root, split)
    loader = _loader(records, config, 0, shuffle=False)
    loss, rows, samples = _evaluate_with_predictions(model, loader, device)
    metrics = _aggregate(rows, split, loss)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_segmentation_artifacts(output_dir, metrics, rows, samples)
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "pipeline": "brisc_2d_segmentation_evaluation",
                "checkpoint": str(checkpoint_path.resolve()),
                "split": split,
                "sample_count": len(records),
                "data_root": str(data_root.resolve()),
                "official_test_split": split == "test",
                "mask_threshold": checkpoint["mask_threshold"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def predict(checkpoint_path: Path, input_dir: Path, output_dir: Path) -> Path:
    from PIL import Image

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = {
        "model": checkpoint["model_config"],
        "image_size": checkpoint["image_size"],
        "mask_threshold": checkpoint["mask_threshold"],
        "device": "auto",
    }
    device = torch.device(runtime_device())
    model = _model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [input_dir] if input_dir.is_file() else sorted(input_dir.iterdir())
    output_count = 0
    with torch.inference_mode():
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            dataset = BRISCClassificationDataset(
                [ClassificationRecord(path, 0)], int(checkpoint["image_size"])
            )
            image, _ = dataset[0]
            prediction = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0]
            binary = (prediction.cpu().numpy() >= 0.5).astype(np.uint8) * 255
            with Image.open(path) as original:
                binary_image = Image.fromarray(binary, mode="L").resize(
                    original.size, Image.Resampling.NEAREST
                )
            binary_image.save(output_dir / f"{path.stem}_mask.png")
            output_count += 1
    if output_count == 0:
        raise ValueError(f"No supported image files found at {input_dir}")
    (output_dir / "prediction_metadata.json").write_text(
        json.dumps(
            {
                "pipeline": "brisc_2d_segmentation",
                "checkpoint": str(checkpoint_path.resolve()),
                "input": str(input_dir.resolve()),
                "prediction_count": output_count,
                "image_size": checkpoint["image_size"],
                "mask_threshold": checkpoint["mask_threshold"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="BRISC 2D segmentation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="3_2d_segmentation/configs/segmentation2d.yaml")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument("--data-root", type=Path)
    evaluate_parser.add_argument("--output-dir", type=Path)
    evaluate_parser.add_argument("--split", choices=("train", "test"), default="test")
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", required=True, type=Path)
    predict_parser.add_argument("--input", required=True, type=Path)
    predict_parser.add_argument("--output-dir", type=Path, default=Path("3_2d_segmentation/outputs/segmentation2d/predictions"))
    args = parser.parse_args()
    if args.command == "train":
        print(train(load_config(args.config)))
    elif args.command == "evaluate":
        data_root = args.data_root or resolve_path("1_data_and_preprocessing/data/BRISC/brisc2025", env_var="BRISC_ROOT")
        output_dir = args.output_dir or args.checkpoint.parent / f"evaluation_{args.split}"
        print(evaluate(args.checkpoint, data_root, output_dir, args.split))
    else:
        print(predict(args.checkpoint, args.input, args.output_dir))
    return 0
