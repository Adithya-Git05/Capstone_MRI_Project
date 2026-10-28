from __future__ import annotations

from typing import Any

import numpy as np

from src.data.brats import REGION_ORDER


def _hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> float | None:
    if not prediction.any() or not target.any():
        return None
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError as error:
        raise RuntimeError("HD95 requires scipy; install the [brats] extra") from error
    prediction_surface = prediction ^ binary_erosion(prediction, border_value=0)
    target_surface = target ^ binary_erosion(target, border_value=0)
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = distance_transform_edt(~prediction_surface, sampling=spacing)
    distances = np.concatenate(
        (distance_to_target[prediction_surface], distance_to_prediction[target_surface])
    )
    return float(np.percentile(distances, 95))


def region_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    region_names: list[str],
    spacing: tuple[float, float, float],
    *,
    compute_hd95: bool = True,
) -> dict[str, dict[str, Any]]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("3D segmentation arrays must match and have shape [regions, X, Y, Z]")
    if prediction.shape[0] != len(region_names):
        raise ValueError("Region count does not match region names")
    if tuple(region_names) != REGION_ORDER:
        raise ValueError(
            f"BraTS region metrics require channel order {REGION_ORDER}; "
            f"received {tuple(region_names)}"
        )
    results = {}
    for index, region_name in enumerate(region_names):
        predicted_region, target_region = prediction[index], target[index]
        intersection = int(np.logical_and(predicted_region, target_region).sum())
        predicted_count = int(predicted_region.sum())
        target_count = int(target_region.sum())
        denominator = predicted_count + target_count
        union = predicted_count + target_count - intersection
        hd95 = _hd95(predicted_region, target_region, spacing) if compute_hd95 else None
        results[region_name] = {
            "dice": float(2 * intersection / denominator) if denominator else 1.0,
            "iou": float(intersection / union) if union else 1.0,
            "hd95_mm": hd95,
            "hd95_valid": hd95 is not None,
            "target_voxels": target_count,
            "predicted_voxels": predicted_count,
        }
    return results
