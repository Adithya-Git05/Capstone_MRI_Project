from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    config["_config_path"] = str(config_path.resolve())
    return config


def resolve_path(
    value: str | Path,
    *,
    env_var: str | None = None,
    root: Path = PROJECT_ROOT,
) -> Path:
    selected = os.environ.get(env_var) if env_var else None
    path = Path(selected or value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def runtime_device(configured: str = "auto") -> str:
    if configured != "auto":
        return configured
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
