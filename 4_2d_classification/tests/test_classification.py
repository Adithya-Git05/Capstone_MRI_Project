from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.classification.engine import _predict
from src.common.metrics import fit_temperature, probability_quality_metrics
from src.data.brisc import (
    ClassificationRecord,
    classification_image_hashes,
    duplicate_aware_split,
)


class ClassificationImprovementTests(unittest.TestCase):
    def test_duplicate_groups_never_cross_internal_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for class_index in range(2):
                for image_index in range(8):
                    path = root / f"class{class_index}_{image_index}.png"
                    Image.new(
                        "L", (12, 8), color=class_index * 80 + image_index
                    ).save(path)
                    records.append(ClassificationRecord(path, class_index))
                    if image_index == 0:
                        duplicate_path = root / f"class{class_index}_duplicate.png"
                        shutil.copyfile(path, duplicate_path)
                        records.append(
                            ClassificationRecord(duplicate_path, class_index)
                        )

            fitting, validation, calibration = duplicate_aware_split(
                records, validation_fraction=0.2, calibration_fraction=0.2, seed=9
            )
            partitions = (fitting, validation, calibration)
            hashes = classification_image_hashes(records)
            hash_sets = [
                {hashes[index] for index in part}
                for part in partitions
            ]
            self.assertFalse(hash_sets[0] & hash_sets[1])
            self.assertFalse(hash_sets[0] & hash_sets[2])
            self.assertFalse(hash_sets[1] & hash_sets[2])
            self.assertEqual(
                set(fitting) | set(validation) | set(calibration),
                set(range(len(records))),
            )
            self.assertFalse(set(fitting) & set(validation))
            self.assertFalse(set(fitting) & set(calibration))
            self.assertFalse(set(validation) & set(calibration))

    def test_temperature_scaling_improves_calibration_nll(self) -> None:
        logits = np.tile(np.asarray([[4.0, 0.0]]), (10, 1))
        targets = np.asarray([0] * 7 + [1] * 3)
        temperature = fit_temperature(logits, targets)
        uncalibrated = probability_quality_metrics(
            targets, torch.softmax(torch.tensor(logits), dim=1).numpy()
        )
        calibrated = probability_quality_metrics(
            targets,
            torch.softmax(torch.tensor(logits) / temperature, dim=1).numpy(),
        )
        self.assertGreater(temperature, 1.0)
        self.assertLess(
            calibrated["negative_log_likelihood"],
            uncalibrated["negative_log_likelihood"],
        )
        self.assertIn("multiclass_brier_score", calibrated)
        self.assertIn("expected_calibration_error", calibrated)
        self.assertEqual(len(calibrated["reliability_bins"]), 15)

    def test_prediction_loss_weights_partial_batch_by_sample_count(self) -> None:
        logits = torch.tensor(
            [[3.0, 0.0], [0.0, 3.0], [0.0, 0.1]], dtype=torch.float32
        )
        targets = torch.tensor([0, 1, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        actual_targets, actual_logits, actual_loss = _predict(
            torch.nn.Identity(), loader, torch.device("cpu")
        )
        expected_loss = torch.nn.functional.cross_entropy(logits, targets).item()
        self.assertEqual(actual_targets.tolist(), targets.tolist())
        np.testing.assert_allclose(actual_logits, logits.numpy())
        self.assertAlmostEqual(actual_loss, expected_loss, places=6)


if __name__ == "__main__":
    unittest.main()
