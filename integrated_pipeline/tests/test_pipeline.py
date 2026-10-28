from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from integrated_pipeline.modules.mri_io import MODALITIES, validate_study, write_uploaded_study
from integrated_pipeline.modules.model_adapters import classify_slice, segment_slice
from integrated_pipeline.modules.report_generator import generate_report
from integrated_pipeline.modules.rag_adapter import validate_rag_evidence
from integrated_pipeline.modules.risk_adapter import risk_prerequisite_status
from integrated_pipeline.modules.slice_extraction import extract_axial_slice, select_slice
from integrated_pipeline.pipeline import run_pipeline
from src.classification.model import BRISCClassifier
from src.data.brats import build_region_masks
from src.data.brisc import CLASS_NAMES
from src.segmentation2d.model import BRISCUNet


class InputValidationTests(unittest.TestCase):
    def test_registered_modalities_preserve_shape_affine_and_spacing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            affine = np.diag([1.2, 1.3, 1.4, 1.0])
            paths = {}
            for modality in MODALITIES:
                path = root / f"{modality}.nii.gz"
                nib.save(nib.Nifti1Image(np.ones((5, 6, 7), dtype=np.float32), affine), path)
                paths[modality] = path
            study = validate_study("synthetic", paths)
            self.assertEqual(study.shape, (5, 6, 7))
            np.testing.assert_allclose(study.affine, affine)
            for actual, expected in zip(study.spacing, (1.2, 1.3, 1.4)):
                self.assertAlmostEqual(actual, expected, places=5)

    def test_rejects_misaligned_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for index, modality in enumerate(MODALITIES):
                affine = np.eye(4)
                if index:
                    affine[0, 3] = 1.0 if index == 3 else 0.0
                path = root / f"{modality}.nii.gz"
                nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), affine), path)
                paths[modality] = path
            with self.assertRaisesRegex(ValueError, "not registered"):
                validate_study("synthetic", paths)

    def test_rejects_non_3d_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            affine = np.eye(4)
            paths = {}
            for modality in MODALITIES:
                path = root / f"{modality}.nii.gz"
                shape = (4, 4, 4, 2) if modality == "t2f" else (4, 4, 4)
                nib.save(nib.Nifti1Image(np.ones(shape), affine), path)
                paths[modality] = path
            with self.assertRaisesRegex(ValueError, "must be a 3D volume"):
                validate_study("synthetic", paths)

    def test_rejects_non_finite_volume_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for modality in MODALITIES:
                data = np.ones((4, 4, 4), dtype=np.float32)
                if modality == "t1c":
                    data[0, 0, 0] = np.nan
                path = root / f"{modality}.nii.gz"
                nib.save(nib.Nifti1Image(data, np.eye(4)), path)
                paths[modality] = path
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_study("synthetic", paths)

    def test_uploaded_modalities_must_be_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Required MRI modality"):
                write_uploaded_study("synthetic", {"t2f": b"file"}, Path(temporary))


class SpatialContractTests(unittest.TestCase):
    def test_brats_region_mapping_preserves_overlapping_channels(self) -> None:
        labels = np.array([[[0, 1, 2, 3, 4]]], dtype=np.int16)
        regions = build_region_masks(labels)
        self.assertEqual(regions.shape, (4, 1, 1, 5))
        self.assertEqual(regions[:, 0, 0, 1].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(regions[:, 0, 0, 3].tolist(), [1.0, 1.0, 1.0, 0.0])

    def test_tumor_slice_selection_tracks_third_axis(self) -> None:
        mask = np.zeros((4, 5, 6), dtype=bool)
        mask[:, :, 4] = True
        self.assertEqual(select_slice(mask, mask.shape), 4)
        self.assertEqual(extract_axial_slice(mask.astype(float), 4).shape, (4, 5))

    def test_empty_mask_selects_middle_slice(self) -> None:
        self.assertEqual(select_slice(np.zeros((2, 3, 8)), (2, 3, 8)), 4)


class IntegrationOutputTests(unittest.TestCase):
    def test_pipeline_rejects_unsafe_study_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "Study identifier"):
            run_pipeline("../case", {}, Path("missing.pt"))

    def test_risk_mapper_requires_explicit_source_config(self) -> None:
        status = risk_prerequisite_status(None, np.zeros((2, 2, 2), dtype=bool))
        self.assertEqual(status["status"], "blocked")
        self.assertIn("SynthSeg anatomical labels", status["reasons"][0])

    def test_risk_preflight_checks_source_files_but_never_claims_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = {}
            for name in ("mri_image", "synthseg_labels", "tumor_mask"):
                path = root / f"{name}.nii.gz"
                path.touch()
                inputs[name] = str(path)
            config = root / "risk.json"
            config.write_text(json.dumps({"paths": inputs}), encoding="utf-8")
            status = risk_prerequisite_status(config, np.ones((2, 2, 2), dtype=bool))
            self.assertEqual(status["status"], "not run")
            self.assertIn("not executed", status["preflight"])

    def test_rag_missing_evidence_is_explicit(self) -> None:
        result = validate_rag_evidence(
            {"answer": {"summary": "No relevant material."}, "retrieved_evidence": [], "citations": []}
        )
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertIn("No retrieved evidence", result["evidence_warning"])

    def test_rag_citation_must_match_retrieved_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present in retrieved evidence"):
            validate_rag_evidence(
                {
                    "retrieved_evidence": [{"chunk_id": "chunk-1", "paper_id": "paper-1"}],
                    "citations": [{"chunk_id": "chunk-2", "paper_id": "paper-2"}],
                }
            )

    def test_pdf_is_generated_from_structured_status(self) -> None:
        document = generate_report(
            {
                "case_id": "synthetic",
                "stages": {
                    "3D segmentation": {"status": "completed"},
                    "risk mapping": {"status": "blocked", "reasons": ["No SynthSeg labels."]},
                },
            }
        )
        self.assertTrue(document.startswith(b"%PDF"))
        self.assertGreater(len(document), 1000)


class ModelAdapterContractTests(unittest.TestCase):
    def test_2d_adapters_return_expected_shapes(self) -> None:
        sample = np.linspace(0, 1, 24 * 32, dtype=np.float32).reshape(24, 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classifier = BRISCClassifier(width=8, dropout=0.25, num_classes=4)
            classifier_path = root / "classifier.pt"
            torch.save(
                {
                    "model_state": classifier.state_dict(),
                    "model_config": {"width": 8, "dropout": 0.25},
                    "image_size": 32,
                    "class_names": list(CLASS_NAMES),
                    "preprocessing": {"resize_mode": "pad"},
                    "temperature": 1.0,
                },
                classifier_path,
            )
            classification = classify_slice(classifier_path, sample)
            self.assertIn(classification["predicted_class"], CLASS_NAMES)
            self.assertAlmostEqual(sum(classification["probabilities"].values()), 1.0, places=5)

            segmenter = BRISCUNet(base_channels=4)
            segmentation_path = root / "segmenter.pt"
            torch.save(
                {
                    "model_state": segmenter.state_dict(),
                    "model_config": {"base_channels": 4},
                    "image_size": 32,
                    "mask_threshold": 0.5,
                },
                segmentation_path,
            )
            mask, metadata = segment_slice(segmentation_path, sample)
            self.assertEqual(mask.shape, sample.shape)
            self.assertTrue(set(np.unique(mask)).issubset({0, 1}))
            self.assertEqual(metadata["image_size"], 32)


if __name__ == "__main__":
    unittest.main()
