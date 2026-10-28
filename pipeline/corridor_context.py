"""Anatomical context for candidate surgical corridors and named structures."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    from .label_mapping import get_label_info
except ImportError:
    from label_mapping import get_label_info

GENERIC_LABELS = {"Brain-Tissue", "Unknown", "Background", "CSF", "WM-hypointensities"}
GENERIC_TRACT_FILES = {"white_matter_tracts"}


def _is_generic_label(name: str, tier: int, weight: float) -> bool:
    return name in GENERIC_LABELS or tier == 0 or weight <= 0.1


def load_tract_masks(tract_dir: Optional[str], shape: Sequence[int]) -> Dict[str, np.ndarray]:
    """Load non-empty tract masks that match the case grid, keyed by file stem."""
    masks: Dict[str, np.ndarray] = {}
    if not tract_dir or not os.path.isdir(tract_dir):
        return masks
    for filename in sorted(os.listdir(tract_dir)):
        if not filename.lower().endswith((".nii", ".nii.gz")):
            continue
        stem = filename.replace(".nii.gz", "").replace(".nii", "")
        mask = nib.load(os.path.join(tract_dir, filename)).get_fdata() > 0
        if tuple(mask.shape) == tuple(shape) and mask.any():
            masks[stem] = mask
    return masks


def nearby_structures(
    centerline_mask: np.ndarray,
    voxel_spacing: Sequence[float],
    labels_data: Optional[np.ndarray] = None,
    tract_masks: Optional[Dict[str, np.ndarray]] = None,
    max_distance_mm: float = 10.0,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """List named structures and tracts within max_distance_mm of a path centerline."""
    if not np.any(centerline_mask):
        return []
    distance = distance_transform_edt(~centerline_mask, sampling=voxel_spacing)
    found: List[Dict[str, Any]] = []
    if labels_data is not None:
        labels = np.round(labels_data).astype(np.int32)
        for label in np.unique(labels):
            if label == 0:
                continue
            name, tier, weight = get_label_info(int(label))
            if _is_generic_label(name, tier, weight):
                continue
            nearest = float(distance[labels == label].min())
            if nearest <= max_distance_mm:
                found.append({
                    "name": name,
                    "kind": "Brain structure",
                    "min_distance_mm": round(nearest, 1),
                })
    for stem, mask in (tract_masks or {}).items():
        if stem in GENERIC_TRACT_FILES:
            continue
        nearest = float(distance[mask].min())
        if nearest <= max_distance_mm:
            found.append({
                "name": stem.replace("_", " ").title(),
                "kind": "White-matter tract",
                "min_distance_mm": round(nearest, 1),
            })
    found.sort(key=lambda item: item["min_distance_mm"])
    return found[:limit]


def build_structure_overlays(
    structure_records: Sequence[Dict[str, Any]],
    labels_data: Optional[np.ndarray],
    tract_masks: Optional[Dict[str, np.ndarray]],
    max_distance_mm: float = 20.0,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """Select named structures near the tumor to draw in the viewer."""
    labels = np.round(labels_data).astype(np.int32) if labels_data is not None else None
    overlays: List[Dict[str, Any]] = []
    for record in structure_records:
        if str(record["structure_type"]).startswith("Generic"):
            continue
        if float(record["tumor_min_distance_mm"]) > max_distance_mm:
            continue
        if record["structure_type"] == "White-Matter Tract":
            stem = str(record["structure_name"]).lower().replace(" ", "_")
            if stem in GENERIC_TRACT_FILES:
                continue
            mask = (tract_masks or {}).get(stem)
        elif labels is not None:
            mask = labels == int(record["label_id"])
        else:
            mask = None
        if mask is None or not np.any(mask):
            continue
        overlays.append({
            "name": record["structure_name"],
            "kind": record["structure_type"],
            "mask": mask,
            "distance_mm": float(record["tumor_min_distance_mm"]),
            "risk_label": record["risk_label"],
            "domain": record["functional_domain"],
            "consequence": record["potential_consequence"],
        })
        if len(overlays) >= limit:
            break
    return overlays
