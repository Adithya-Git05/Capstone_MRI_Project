from __future__ import annotations

from functools import lru_cache
import pickle
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


@lru_cache(maxsize=8)
def _load_checkpoint(path: str, modified_ns: int, size: int) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise ValueError(f"Checkpoint could not be safely loaded: {path}") from error
    if not isinstance(value, dict) or not isinstance(value.get("model_state"), dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return value


def _checkpoint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved}")
    details = resolved.stat()
    return _load_checkpoint(str(resolved), details.st_mtime_ns, details.st_size)


def _to_pil(
    slice_image: np.ndarray, percentile_window: bool = False
) -> Image.Image:
    if slice_image.ndim != 2:
        raise ValueError(f"Expected a 2D image slice, got shape {slice_image.shape}.")
    if not np.isfinite(slice_image).all():
        raise ValueError("MRI slice contains non-finite voxel values.")
    finite = np.asarray(slice_image, dtype=np.float32)
    if percentile_window:
        nonzero = finite != 0
        if nonzero.any():
            low, high = np.percentile(finite[nonzero], (1, 99))
        else:
            low, high = 0.0, 0.0
        if high > low:
            finite = np.clip((finite - low) / (high - low), 0.0, 1.0) * 255.0
        else:
            finite = np.zeros_like(finite)
    finite = np.clip(finite, 0.0, 255.0)
    return Image.fromarray(np.round(finite).astype(np.uint8), mode="L")


def _tensor(
    slice_image: np.ndarray,
    size: int,
    pad: bool,
    normalize_to_minus_one: bool,
    percentile_window: bool = False,
) -> torch.Tensor:
    image = _to_pil(slice_image, percentile_window=percentile_window)
    dimensions = (size, size)
    if pad:
        image = ImageOps.pad(image, dimensions, method=Image.Resampling.BILINEAR, color=0)
    else:
        image = image.resize(dimensions, Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values.copy()).unsqueeze(0).unsqueeze(0)
    return tensor * 2.0 - 1.0 if normalize_to_minus_one else tensor


@lru_cache(maxsize=4)
def _load_classifier(
    checkpoint_path: str, modified_ns: int, size: int
) -> tuple[torch.nn.Module, dict[str, Any]]:
    from src.classification.model import BRISCClassifier
    from src.data.brisc import CLASS_NAMES

    checkpoint = _checkpoint(Path(checkpoint_path))
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Classifier checkpoint class order does not match BRISC.")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Classifier checkpoint is missing a valid model_config.")
    model = BRISCClassifier(
        width=int(model_config.get("width", 32)),
        dropout=float(model_config.get("dropout", 0.25)),
        num_classes=len(CLASS_NAMES),
    )
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise ValueError("Classifier checkpoint weights do not match BRISCClassifier.") from error
    model.eval()
    return model, checkpoint


@lru_cache(maxsize=4)
def _load_segmenter(
    checkpoint_path: str, modified_ns: int, size: int
) -> tuple[torch.nn.Module, dict[str, Any], bool]:
    checkpoint = _checkpoint(Path(checkpoint_path))
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Segmentation checkpoint is missing a valid model_config.")
    is_attention_model = any(
        key.startswith("att3.") for key in checkpoint["model_state"]
    )
    if is_attention_model:
        from src.segmentation2d.model_attention import BRISCAttentionUNet

        model = BRISCAttentionUNet(
            base_channels=int(config.get("base_channels", 16))
        )
    else:
        from src.segmentation2d.model import BRISCUNet

        model = BRISCUNet(base_channels=int(config.get("base_channels", 16)))
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise ValueError("Segmentation checkpoint weights do not match its architecture.") from error
    model.eval()
    return model, checkpoint, is_attention_model


def classify_slice(
    checkpoint_path: Path,
    slice_image: np.ndarray,
    percentile_window: bool = False,
) -> dict[str, Any]:
    from src.data.brisc import CLASS_NAMES

    resolved = checkpoint_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved}")
    details = resolved.stat()
    model, checkpoint = _load_classifier(
        str(resolved), details.st_mtime_ns, details.st_size
    )
    size = int(checkpoint["image_size"])
    resize_mode = checkpoint.get("preprocessing", {}).get("resize_mode", "stretch")
    image = _tensor(
        slice_image,
        size,
        pad=resize_mode == "pad",
        normalize_to_minus_one=True,
        percentile_window=percentile_window,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(image)[0]
        probabilities = torch.softmax(logits, dim=0).numpy()
    raw_scores = logits.numpy()
    prediction = int(probabilities.argmax())
    return {
        "predicted_class": CLASS_NAMES[prediction],
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
        "scores": {name: float(raw_scores[i]) for i, name in enumerate(CLASS_NAMES)},
        "score_interpretation": "uncalibrated softmax of model logits; not a clinical probability",
        "checkpoint": str(checkpoint_path.resolve()),
        "architecture": "BRISCClassifier",
        "class_names": list(CLASS_NAMES),
        "image_size": size,
        "resize_mode": resize_mode,
        "inference_time_seconds": time.perf_counter() - started,
        "preprocessing": (
            (
                "grayscale; per-slice nonzero 1st-99th percentile intensity window "
                "(NIfTI domain adaptation); then normalized to [-1,1]"
                if percentile_window
                else "grayscale; 8-bit input scaled to [0,1], then normalized to [-1,1]"
            )
        ),
        "interpretation": "BRISC slice-level model output; not a confirmed diagnosis.",
    }


def segment_slice(
    checkpoint_path: Path,
    slice_image: np.ndarray,
    percentile_window: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = checkpoint_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved}")
    details = resolved.stat()
    model, checkpoint, is_attention_model = _load_segmenter(
        str(resolved), details.st_mtime_ns, details.st_size
    )
    size = int(checkpoint["image_size"])
    tensor = _tensor(
        slice_image,
        size,
        pad=False,
        normalize_to_minus_one=True,
        percentile_window=percentile_window,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].numpy()
    inference_time = time.perf_counter() - started
    prediction_threshold = 0.5
    binary = (probability >= prediction_threshold).astype(np.uint8)
    restored = Image.fromarray(binary * 255).resize(
        (slice_image.shape[1], slice_image.shape[0]), Image.Resampling.NEAREST
    )
    return (np.asarray(restored) > 0).astype(np.uint8), {
        "checkpoint": str(checkpoint_path.resolve()),
        "architecture": "BRISCAttentionUNet" if is_attention_model else "BRISCUNet",
        "classes": {"0": "background", "1": "tumor foreground"},
        "image_size": size,
        "resize_mode": "stretch",
        "prediction_threshold": prediction_threshold,
        "ground_truth_mask_threshold": int(checkpoint["mask_threshold"]),
        "inference_time_seconds": inference_time,
        "preprocessing": (
            (
                "grayscale; per-slice nonzero 1st-99th percentile intensity window "
                "(NIfTI domain adaptation), then normalized to [-1,1]"
                if percentile_window
                else "grayscale; 8-bit input scaled to [0,1], then normalized to [-1,1]"
            )
        ),
    }
