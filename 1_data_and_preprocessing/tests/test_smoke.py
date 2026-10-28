from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src.classification.model import BRISCClassifier
from src.common.config import PROJECT_ROOT
from src.common.metrics import (
    binary_segmentation_metrics,
    classification_metrics,
)
from src.data.brats import (
    BraTSVolumeDataset,
    grouped_kfold,
    grouped_train_validation_split,
    list_cases,
)
from src.data.brisc import (
    BRISCClassificationDataset,
    BRISCSegmentationDataset,
    CLASS_NAMES,
    classification_records,
    segmentation_records,
    stratified_split,
)
from src.segmentation2d.engine import _loss as segmentation_loss
from src.segmentation2d.model import BRISCUNet
from src.segmentation3d.metrics import region_metrics
from src.segmentation3d.nnunet_adapter import _assert_region_mappings, _read_source_config


class PipelineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.brisc_root = PROJECT_ROOT / "data" / "BRISC" / "brisc2025"
        cls.brats_root = PROJECT_ROOT / "data" / "BraTS" / "training_data_additional"

    def test_brisc_classification_dataset_and_model_step(self) -> None:
        records = classification_records(self.brisc_root, "train")
        self.assertEqual(len(records), 5000)
        self.assertEqual({record.label for record in records}, set(range(4)))
        train_indices, validation_indices = stratified_split(records, 0.15, 123)
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertEqual(len(train_indices) + len(validation_indices), len(records))

        image, label = BRISCClassificationDataset([records[0]], image_size=64)[0]
        self.assertEqual(tuple(image.shape), (1, 64, 64))
        logits = BRISCClassifier(width=4, dropout=0.0)(image.unsqueeze(0))
        self.assertEqual(tuple(logits.shape), (1, len(CLASS_NAMES)))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([label]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_brisc_segmentation_pair_and_model_step(self) -> None:
        records = segmentation_records(self.brisc_root, "train")
        self.assertEqual(len(records), 3933)
        image, mask = BRISCSegmentationDataset(records[:1], image_size=64)[0]
        self.assertEqual(tuple(image.shape), (1, 64, 64))
        self.assertEqual(tuple(mask.shape), (1, 64, 64))
        self.assertTrue(set(torch.unique(mask).tolist()).issubset({0.0, 1.0}))
        model = BRISCUNet(base_channels=4)
        logits = model(image.unsqueeze(0))
        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))
        loss = segmentation_loss(logits, mask.unsqueeze(0))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_classification_metrics_output_shape(self) -> None:
        targets = np.asarray([0, 1, 2, 3, 0, 1])
        probabilities = np.asarray(
            [
                [0.9, 0.03, 0.04, 0.03],
                [0.05, 0.8, 0.1, 0.05],
                [0.05, 0.1, 0.8, 0.05],
                [0.05, 0.05, 0.1, 0.8],
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
            ]
        )
        report = classification_metrics(targets, probabilities, list(CLASS_NAMES))
        self.assertEqual(report["sample_count"], 6)
        self.assertEqual(report["confusion_matrix"][0][0], 2)
        self.assertEqual(len(report["per_class"]), 4)
        self.assertIsNotNone(report["per_class"][0]["auroc_ovr"])
        self.assertEqual(report["accuracy"], 1.0)

    def test_segmentation_metrics_empty_and_overlap(self) -> None:
        target = np.asarray([[False, True], [False, True]])
        self.assertEqual(
            binary_segmentation_metrics(target, target),
            {"dice": 1.0, "iou": 1.0},
        )
        self.assertEqual(
            binary_segmentation_metrics(np.zeros_like(target), np.zeros_like(target)),
            {"dice": 1.0, "iou": 1.0},
        )

    def test_brats_region_metrics_and_hd95_validity(self) -> None:
        target = np.zeros((4, 8, 8, 8), dtype=bool)
        target[0, 2:4, 2:4, 2:4] = True
        target[1, 2:4, 2:4, 2:4] = True
        target[2, 2:4, 2:4, 2:4] = True
        prediction = target.copy()
        prediction[3, 1, 1, 1] = True
        scores = region_metrics(
            prediction, target, ["WT", "TC", "ET", "RC"], (1.0, 1.0, 1.0)
        )
        self.assertEqual(scores["WT"]["dice"], 1.0)
        self.assertEqual(scores["WT"]["hd95_mm"], 0.0)
        self.assertFalse(scores["RC"]["hd95_valid"])
        self.assertIsNone(scores["RC"]["hd95_mm"])

    def test_braTS_nnunet_regions_use_approved_composites(self) -> None:
        config = {
            "dataset_config": "legacy/BraTS_Challenge/nnUNet_hyperparameters/dataset.json",
            "label_scheme": "brats_post_treatment_composites_v1",
            "region_order": ["WT", "TC", "ET", "RC"],
            "region_labels": {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3], "RC": [4]},
        }
        _assert_region_mappings(config, _read_source_config(config))

    def test_brats_grouped_split_and_volume_loader(self) -> None:
        cases = list_cases(self.brats_root)
        self.assertEqual(len(cases), 271)
        train, validation = grouped_train_validation_split(cases, 0.2, 123)
        train_groups = {case.patient_group for case in train}
        validation_groups = {case.patient_group for case in validation}
        self.assertFalse(train_groups & validation_groups)
        self.assertEqual(len(train) + len(validation), len(cases))
        folds = grouped_kfold(cases, 5, 123)
        for fold in folds:
            self.assertFalse(set(fold["train"]) & set(fold["val"]))
            train_ids = {case.case_id: case.patient_group for case in cases if case.case_id in fold["train"]}
            validation_ids = {case.case_id: case.patient_group for case in cases if case.case_id in fold["val"]}
            self.assertFalse(set(train_ids.values()) & set(validation_ids.values()))

        sample = BraTSVolumeDataset(cases[:1], patch_size=(32, 32, 32))[0]
        self.assertEqual(tuple(sample["image"].shape), (4, 32, 32, 32))
        self.assertEqual(tuple(sample["label"].shape), (4, 32, 32, 32))
        self.assertEqual(sample["patient_group"], cases[0].patient_group)
        self.assertTrue(torch.isfinite(sample["image"]).all())


if __name__ == "__main__":
    unittest.main()
