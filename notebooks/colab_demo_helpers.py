"""Small, auditable helpers used by Capstone_Full_Demo.ipynb."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_nifti_study(paths: dict[str, str | Path]) -> dict[str, Any]:
    """Validate explicit T1/T1ce/T2/FLAIR mapping and shared voxel geometry."""
    import nibabel as nib

    required = ("T1", "T1ce", "T2", "FLAIR")
    missing = [name for name in required if not paths.get(name)]
    if missing:
        raise ValueError(f"Missing required modalities: {missing}")
    images = {name: nib.load(str(paths[name])) for name in required}
    reference = images["T1"]
    if len(reference.shape) != 3:
        raise ValueError(f"T1 must be 3D, got {reference.shape}")
    rows = {}
    for name, image in images.items():
        if len(image.shape) != 3 or image.shape != reference.shape:
            raise ValueError(f"{name} shape {image.shape} does not match T1 {reference.shape}")
        if not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=1e-5):
            raise ValueError(f"{name} affine does not match T1; registration/resampling is required")
        data = image.get_fdata(dtype=np.float32, caching="unchanged")
        if not np.isfinite(data).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        nonzero = data != 0
        if not nonzero.any() or float(data[nonzero].std()) < 1e-8:
            raise ValueError(f"{name} has no usable nonzero intensity variation")
        rows[name] = {"path": str(paths[name]), "shape": list(image.shape),
                      "spacing_mm": list(map(float, image.header.get_zooms()[:3])),
                      "sha256": sha256(paths[name])}
    return {"status": "Completed", "modalities": rows,
            "affine": reference.affine.tolist(),
            "orientation": list(nib.aff2axcodes(reference.affine)),
            "note": "Geometry is already aligned; no transform was applied."}


def validate_brat_checkpoint(path: str | Path) -> dict[str, Any]:
    """Validate the model-card checkpoint structure without trusting embedded objects."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint dictionary containing state_dict")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("state_dict is empty or invalid")
    head = [(key, tuple(value.shape)) for key, value in state.items()
            if key.endswith("out.conv.conv.weight") or "out.conv" in key]
    return {"status": "Checkpoint container validated; architecture compatibility pending strict load",
            "keys": list(checkpoint), "state_tensor_count": len(state),
            "output_head_candidates": head, "sha256": sha256(path)}


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def make_artifact_zip(root: str | Path, output: str | Path) -> Path:
    root, output = Path(root).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_file() and path.resolve() != output:
                archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
    return output

