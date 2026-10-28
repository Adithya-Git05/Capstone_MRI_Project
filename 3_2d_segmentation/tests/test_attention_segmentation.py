from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.common.config import PROJECT_ROOT
from src.data.brisc import BRISCSegmentationDataset, SegmentationRecord
from src.segmentation2d.engine_attention import (
    _aggregate,
    _create_evaluation_dir,
    _evaluate_with_predictions,
    _save_evaluation_artifacts,
    _sha256_file,
)
from src.segmentation2d.model_attention import BRISCAttentionUNet


EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation2d"
    / "02_attention_boundary_unet_import_20261019_1209"
)


class AttentionSegmentationTests(unittest.TestCase):
    def test_archived_checkpoint_loads_strictly_with_expected_attention_shapes(self) -> None:
        checkpoint_path = EXPERIMENT_DIR / "best.pt"
        self.assertTrue(checkpoint_path.is_file())
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        self.assertEqual(checkpoint["model_config"]["base_channels"], 16)
        model = BRISCAttentionUNet(
            base_channels=checkpoint["model_config"]["base_channels"]
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        self.assertEqual(
            tuple(checkpoint["model_state"]["att3.skip_proj.weight"].shape),
            (32, 64, 1, 1),
        )
        self.assertEqual(
            tuple(checkpoint["model_state"]["att2.skip_proj.weight"].shape),
            (16, 32, 1, 1),
        )
        self.assertEqual(
            tuple(checkpoint["model_state"]["att1.skip_proj.weight"].shape),
            (8, 16, 1, 1),
        )
        self.assertEqual(
            _sha256_file(checkpoint_path),
            "282f1499be25cef06bc20d828317ab2cef17771e26891652466e0b207f1dccab",
        )

    def test_attention_model_forward_shape_and_finite_logits(self) -> None:
        model = BRISCAttentionUNet(base_channels=16).eval()
        with torch.inference_mode():
            logits = model(torch.zeros(1, 1, 256, 256))
        self.assertEqual(tuple(logits.shape), (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(logits).all())

    def test_synthetic_evaluation_metrics_and_artifact_saving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for index in range(2):
                image_path = root / f"image_{index}.png"
                mask_path = root / f"image_{index}_mask.png"
                image = np.full((40, 48), 120 + index * 20, dtype=np.uint8)
                mask = np.zeros((40, 48), dtype=np.uint8)
                mask[10:30, 12:36] = 255
                Image.fromarray(image).save(image_path)
                Image.fromarray(mask).save(mask_path)
                records.append(SegmentationRecord(image_path, mask_path))
            dataset = BRISCSegmentationDataset(records, image_size=32, mask_threshold=128)
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            model = BRISCAttentionUNet(base_channels=2).eval()
            loss, rows, samples = _evaluate_with_predictions(
                model, loader, torch.device("cpu")
            )
            metrics = _aggregate(rows, "test", loss)
            self.assertEqual(metrics["sample_count"], 2)
            self.assertIn("global_foreground_dice", metrics)
            self.assertIn("global_foreground_iou", metrics)
            self.assertEqual(len(rows), 2)
            output = root / "evaluation"
            output.mkdir()
            _save_evaluation_artifacts(output, metrics, rows, samples)
            self.assertTrue((output / "test_metrics.json").is_file())
            self.assertTrue((output / "per_image_metrics.csv").is_file())
            self.assertTrue((output / "test_metric_distributions.png").is_file())
            self.assertTrue((output / "representative_test_overlays.png").is_file())
            saved = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["split"], "test")
            with (output / "per_image_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_output_directory_creation_never_overwrites_previous_results(self) -> None:
        checkpoint_path = EXPERIMENT_DIR / "best.pt"
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "existing"
            existing.mkdir()
            sentinel = existing / "sentinel.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _create_evaluation_dir(checkpoint_path, "test", existing)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
