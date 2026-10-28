from __future__ import annotations

import numpy as np


def select_slice(mask: np.ndarray | None, volume_shape: tuple[int, int, int]) -> int:
    """Select the axial RAS slice with most tumor voxels, or the middle if mask is empty."""
    if mask is None:
        return volume_shape[2] // 2
    if mask.shape != volume_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match MRI shape {volume_shape}.")
    counts = np.count_nonzero(mask, axis=(0, 1))
    return int(np.argmax(counts)) if counts.any() else volume_shape[2] // 2


def extract_axial_slice(volume: np.ndarray, index: int) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim}D.")
    if not 0 <= index < volume.shape[2]:
        raise IndexError(f"Axial slice {index} is outside [0, {volume.shape[2] - 1}].")
    return np.asarray(volume[:, :, index], dtype=np.float32)
