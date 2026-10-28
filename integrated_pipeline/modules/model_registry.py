from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "model_registry.json"
MODEL_IDS = ("segmentation_2d", "classification_2d", "segmentation_3d")


def load_model_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, dict[str, Any]]:
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read model registry: {path}") from error
    if not isinstance(registry, dict) or not isinstance(registry.get("models"), dict):
        raise ValueError("Model registry must contain a 'models' object.")

    models = registry["models"]
    missing = sorted(set(MODEL_IDS) - set(models))
    if missing:
        raise ValueError(f"Model registry is missing entries: {', '.join(missing)}.")
    validated: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        model = models[model_id]
        if not isinstance(model, dict):
            raise ValueError(f"Registry entry '{model_id}' must be an object.")
        if model.get("task") != "segmentation" and model_id != "classification_2d":
            raise ValueError(f"Registry entry '{model_id}' has an invalid task.")
        if model_id == "classification_2d" and model.get("task") != "classification":
            raise ValueError("Registry entry 'classification_2d' must be a classifier.")
        expected_dimension = "3D" if model_id == "segmentation_3d" else "2D"
        if model.get("dimensionality") != expected_dimension:
            raise ValueError(
                f"Registry entry '{model_id}' must declare {expected_dimension} dimensionality."
            )
        checkpoint = model.get("checkpoint")
        if checkpoint is not None:
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                raise ValueError(f"Registry entry '{model_id}' has an invalid checkpoint path.")
            relative_path = Path(checkpoint)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Registry checkpoint for '{model_id}' must be a safe project-relative path."
                )
            model["resolved_checkpoint"] = (PROJECT_ROOT / relative_path).resolve()
        elif model.get("status") != "not_configured":
            raise ValueError(
                f"Registry entry '{model_id}' needs a checkpoint or not_configured status."
            )
        validated[model_id] = model
    return validated


def verify_checkpoint_integrity(path: Path, model: dict[str, Any]) -> str:
    """Verify the manifest hash for the selected checkpoint; label overrides explicitly."""
    resolved = path.expanduser().resolve()
    if resolved != model.get("resolved_checkpoint"):
        return "override not covered by the selected-checkpoint manifest"
    expected = model.get("checkpoint_integrity")
    if not isinstance(expected, dict):
        raise ValueError("Selected checkpoint has no recorded integrity metadata.")
    if not resolved.is_file():
        raise FileNotFoundError(f"Selected model checkpoint not found: {resolved}")
    if resolved.stat().st_size != expected.get("size_bytes"):
        raise ValueError("Selected checkpoint size does not match the recorded manifest.")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != str(expected.get("sha256", "")).lower():
        raise ValueError("Selected checkpoint SHA-256 does not match the recorded manifest.")
    return "verified against checkpoint manifest"
