from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.config import load_config, resolve_path
from src.classification import engine as classification_engine
from src.segmentation2d import engine as segmentation_engine


EXPERIMENTS = (
    ("classification", "resnet50", "4_2d_classification/configs/classification_resnet50.yaml"),
    (
        "classification",
        "efficientnet_b0",
        "4_2d_classification/configs/classification_efficientnet_b0.yaml",
    ),
    ("segmentation2d", "resnet50", "3_2d_segmentation/configs/segmentation2d_resnet50.yaml"),
    (
        "segmentation2d",
        "efficientnet_b0",
        "3_2d_segmentation/configs/segmentation2d_efficientnet_b0.yaml",
    ),
)


def run_experiments() -> Path:
    summary: dict[str, Any] = {
        "device": "cpu",
        "initialization": "random; torchvision weights=None",
        "experiments": [],
    }
    errors: list[str] = []

    for task, architecture, config_path in EXPERIMENTS:
        record: dict[str, Any] = {
            "task": task,
            "architecture": architecture,
            "config": config_path,
            "status": "running",
        }
        summary["experiments"].append(record)
        print(f"START {task}/{architecture} using {config_path}", flush=True)
        try:
            config = load_config(config_path)
            if task == "classification":
                run_dir = classification_engine.train(config)
                evaluation_dir = classification_engine.evaluate(
                    run_dir / "best.pt",
                    resolve_path(config["data_root"], env_var="BRISC_ROOT"),
                    run_dir / "evaluation_test",
                    "test",
                )
                metrics_path = evaluation_dir / "metrics.json"
            else:
                run_dir = segmentation_engine.train(config)
                evaluation_dir = segmentation_engine.evaluate(
                    run_dir / "best.pt",
                    resolve_path(config["data_root"], env_var="BRISC_ROOT"),
                    run_dir / "evaluation_test",
                    "test",
                )
                metrics_path = evaluation_dir / "test_metrics.json"
            record.update(
                status="completed",
                run_dir=str(run_dir.resolve()),
                test_evaluation_dir=str(evaluation_dir.resolve()),
                test_metrics=str(metrics_path.resolve()),
            )
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            errors.append(f"{task}/{architecture}: {record['error']}")
            output_root = Path(
                "4_2d_classification/outputs/classification"
                if task == "classification"
                else "3_2d_segmentation/outputs/segmentation2d"
            )
            output_root.mkdir(parents=True, exist_ok=True)
            print(f"FAILED {task}/{architecture}: {record['error']}", flush=True)

    summary_path = Path("3_2d_segmentation/outputs/backbone_experiments_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary["status"] = "failed" if errors else "completed"
    summary["errors"] = errors
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SUMMARY={summary_path.resolve()}", flush=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    return summary_path


if __name__ == "__main__":
    run_experiments()
