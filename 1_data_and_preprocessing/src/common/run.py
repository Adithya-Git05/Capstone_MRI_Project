from __future__ import annotations

import json
import logging
import platform
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy
import torch
import yaml


def create_run_dir(output_root: Path, run_name: str) -> Path:
    run_dir = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("brain_tumor_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_run_metadata(
    run_dir: Path,
    config: dict[str, Any],
    *,
    pipeline: str,
    split_details: dict[str, Any],
) -> None:
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(clean_config, sort_keys=True), encoding="utf-8"
    )
    package_versions = {
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    for package in ("Pillow", "PyYAML", "nibabel", "monai", "nnunetv2", "scipy"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            continue
    metadata = {
        "pipeline": pipeline,
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": config.get("seed"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions,
        "device": config.get("device"),
        "split_details": split_details,
        "checkpoint": None,
        "metrics": None,
        "source_config": config.get("_config_path"),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )


def update_run_metadata(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(updates)
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
