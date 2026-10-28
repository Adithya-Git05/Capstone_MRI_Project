from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


def _checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise ValueError(f"Checkpoint could not be safely loaded: {path}") from error
    if not isinstance(value, dict) or not isinstance(value.get("model_state"), dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return value


def _to_pil(slice_image: np.ndarray) -> Image.Image:
    if slice_image.ndim != 2:
        raise ValueError(f"Expected a 2D image slice, got shape {slice_image.shape}.")
    if not np.isfinite(slice_image).all():
        raise ValueError("MRI slice contains non-finite voxel values.")
    finite = np.asarray(slice_image, dtype=np.float32)
    nonzero = finite != 0
    if nonzero.any():
        low, high = np.percentile(finite[nonzero], (1, 99))
    else:
        low, high = 0.0, 0.0
    if high > low:
        finite = np.clip((finite - low) / (high - low), 0.0, 1.0)
    else:
        finite = np.zeros_like(finite)
    return Image.fromarray(np.round(finite * 255).astype(np.uint8), mode="L")


def _tensor(
    slice_image: np.ndarray, size: int, pad: bool, normalize_to_minus_one: bool
) -> torch.Tensor:
    image = _to_pil(slice_image)
    dimensions = (size, size)
    if pad:
        image = ImageOps.pad(image, dimensions, method=Image.Resampling.BILINEAR, color=0)
    else:
        image = image.resize(dimensions, Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values.copy()).unsqueeze(0).unsqueeze(0)
    return tensor * 2.0 - 1.0 if normalize_to_minus_one else tensor


def classify_slice(checkpoint_path: Path, slice_image: np.ndarray) -> dict[str, Any]:
    from src.classification.model import BRISCClassifier
    from src.data.brisc import CLASS_NAMES

    checkpoint = _checkpoint(checkpoint_path)
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Classifier checkpoint class order does not match BRISC.")
    model_config = checkpoint["model_config"]
    model = BRISCClassifier(
        width=int(model_config.get("width", 32)),
        dropout=float(model_config.get("dropout", 0.25)),
        num_classes=len(CLASS_NAMES),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    size = int(checkpoint["image_size"])
    resize_mode = checkpoint.get("preprocessing", {}).get("resize_mode", "stretch")
    image = _tensor(
        slice_image, size, pad=resize_mode == "pad", normalize_to_minus_one=True
    )
    temperature = float(checkpoint.get("temperature", 1.0))
    with torch.inference_mode():
        probabilities = torch.softmax(model(image) / temperature, dim=1)[0].numpy()
    prediction = int(probabilities.argmax())
    return {
        "predicted_class": CLASS_NAMES[prediction],
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
        "checkpoint": str(checkpoint_path.resolve()),
        "image_size": size,
        "resize_mode": resize_mode,
        "preprocessing": (
            "grayscale; nonzero MRI 1st-99th percentile mapped to [0,1], "
            "then normalized to [-1,1]"
        ),
        "interpretation": "BRISC slice-level model output; not a confirmed diagnosis.",
    }


def segment_slice(
    checkpoint_path: Path, slice_image: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint = _checkpoint(checkpoint_path)
    config = checkpoint["model_config"]
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
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    size = int(checkpoint["image_size"])
    tensor = _tensor(
        slice_image,
        size,
        pad=False,
        normalize_to_minus_one=is_attention_model,
    )
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].numpy()
    prediction_threshold = 0.5
    binary = (probability >= prediction_threshold).astype(np.uint8)
    restored = Image.fromarray(binary * 255).resize(
        (slice_image.shape[1], slice_image.shape[0]), Image.Resampling.NEAREST
    )
    return (np.asarray(restored) > 0).astype(np.uint8), {
        "checkpoint": str(checkpoint_path.resolve()),
        "image_size": size,
        "resize_mode": "stretch",
        "architecture": "BRISCAttentionUNet" if is_attention_model else "BRISCUNet",
        "prediction_threshold": prediction_threshold,
        "ground_truth_mask_threshold": int(checkpoint["mask_threshold"]),
        "preprocessing": (
            "grayscale; nonzero MRI 1st-99th percentile mapped to [0,1], "
            "then normalized to [-1,1]"
            if is_attention_model
            else "grayscale; nonzero MRI 1st-99th percentile mapped to [0,1]"
        ),
    }
