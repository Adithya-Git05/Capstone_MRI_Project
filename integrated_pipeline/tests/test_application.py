from __future__ import annotations

import hashlib
from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from PIL import Image
import requests
from streamlit.testing.v1 import AppTest

from integrated_pipeline.app import _empty_result, _format_cited_answer, _run_analysis
from integrated_pipeline.modules import rag_adapter
from integrated_pipeline.modules.model_registry import load_model_registry
from integrated_pipeline.modules.model_registry import verify_checkpoint_integrity
from integrated_pipeline.modules.model_adapters import _to_pil
from integrated_pipeline.modules.mri_io import (
    load_mri_upload,
    select_uploaded_slice,
    validate_uploaded_study,
)
from integrated_pipeline.modules.report_generator import (
    generate_markdown_report,
    generate_pdf_report,
    generate_report,
)
from integrated_pipeline.modules.rag_adapter import build_patient_context


class UploadedMRIInputTests(unittest.TestCase):
    def test_loads_jpeg_as_finite_grayscale_image(self) -> None:
        image = Image.fromarray(
            np.arange(48, dtype=np.uint8).reshape(6, 8), mode="L"
        )
        buffer = BytesIO()
        image.save(buffer, format="JPEG")

        loaded = load_mri_upload(buffer.getvalue(), "scan.jpg")

        self.assertEqual(loaded.data.shape, (6, 8))
        self.assertEqual(loaded.metadata["format"], "JPG")
        self.assertEqual(select_uploaded_slice(loaded, 0).shape, (6, 8))

    def test_loads_nifti_and_selects_requested_axial_slice(self) -> None:
        data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        affine = np.diag([-1.2, -1.3, 1.4, 1.0])
        image = nib.Nifti1Image(data, affine)
        payload = image.to_bytes()

        loaded = load_mri_upload(payload, "volume.nii")

        self.assertEqual(loaded.data.shape, (4, 5, 6))
        self.assertEqual(loaded.metadata["orientation"], "RAS")
        np.testing.assert_array_equal(
            select_uploaded_slice(loaded, 3), data[::-1, ::-1, 3]
        )

    def test_rejects_bad_image_and_invalid_slice_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a readable"):
            load_mri_upload(b"not an image", "scan.png")
        loaded = load_mri_upload(
            nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)).to_bytes(),
            "volume.nii",
        )
        with self.assertRaises(IndexError):
            select_uploaded_slice(loaded, 2)

    def test_loads_compressed_nifti_and_rejects_4d_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "volume.nii.gz"
            nib.save(
                nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.float32), np.eye(4)),
                path,
            )
            compressed_payload = path.read_bytes()
        loaded = load_mri_upload(compressed_payload, "volume.nii.gz")
        self.assertEqual(loaded.data.shape, (3, 4, 5))

        payload_4d = nib.Nifti1Image(
            np.ones((3, 4, 5, 2), dtype=np.float32), np.eye(4)
        ).to_bytes()
        with self.assertRaisesRegex(ValueError, "must be a 3D volume"):
            load_mri_upload(payload_4d, "multi-volume.nii")

    def test_png_intensity_conversion_matches_8_bit_training_scale(self) -> None:
        normalized = np.asarray(
            _to_pil(np.array([[0.0, 127.5, 255.0]], dtype=np.float32))
        )
        np.testing.assert_array_equal(normalized, np.array([[0, 128, 255]], dtype=np.uint8))

    def test_validates_four_registered_uploaded_modalities(self) -> None:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        payloads = {
            modality: nib.Nifti1Image(
                np.full((4, 5, 6), index + 1, dtype=np.float32), affine
            ).to_bytes()
            for index, modality in enumerate(("t2f", "t1n", "t1c", "t2w"))
        }

        study = validate_uploaded_study("case-01", payloads)

        self.assertEqual(study["modalities"], ["t2f", "t1n", "t1c", "t2w"])
        self.assertEqual(study["shape"], (4, 5, 6))
        self.assertEqual(study["orientation"], "RAS")
        self.assertEqual(study["voxel_spacing_mm"], (1.0, 1.0, 1.0))

    def test_multimodal_upload_rejects_missing_or_misaligned_sequences(self) -> None:
        with self.assertRaisesRegex(ValueError, "was not supplied"):
            validate_uploaded_study("case-01", {"t2f": b"not-written"})

        payloads = {}
        for index, modality in enumerate(("t2f", "t1n", "t1c", "t2w")):
            affine = np.eye(4)
            if modality == "t2w":
                affine[0, 3] = 2
            payloads[modality] = nib.Nifti1Image(
                np.ones((4, 5, 6), dtype=np.float32), affine
            ).to_bytes()
        with self.assertRaisesRegex(ValueError, "not registered"):
            validate_uploaded_study("case-01", payloads)

    def test_multimodal_upload_rejects_unsafe_study_identifier(self) -> None:
        with self.assertRaisesRegex(ValueError, "path separators"):
            validate_uploaded_study("../case", {})


class RegistryAndReportTests(unittest.TestCase):
    def test_chat_answer_is_plain_text_with_verified_knowledge_base_citations(self) -> None:
        formatted = _format_cited_answer(
            {
                "answer": {
                    "summary": "The supplied MRI classification is experimental [P1_C001].",
                },
                "citations": [
                    {
                        "chunk_id": "P1_C001",
                        "title": "MRI Study",
                        "status": "verified",
                    },
                    {
                        "chunk_id": "P2_C001",
                        "title": "Unverified study",
                        "status": "unverified",
                    },
                ],
                "retrieved_evidence": [
                    {
                        "chunk_id": "P1_C001",
                        "title": "MRI Study",
                        "passage": "This passage is not shown in the chat.",
                        "metadata": {"authors": ["A. Author"], "year": 2025},
                    },
                    {
                        "chunk_id": "P2_C001",
                        "title": "Unverified study",
                        "passage": "Unverified source text.",
                    },
                ],
            }
        )

        self.assertIn("experimental [1]", formatted)
        self.assertIn("[1] MRI Study. A. Author. 2025.", formatted)
        self.assertNotIn("This passage is not shown", formatted)
        self.assertNotIn("Unverified study", formatted)

    def test_chat_page_renders_without_an_mri_upload(self) -> None:
        app_path = Path(__file__).resolve().parents[2] / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=20)

        self.assertEqual(app.exception, [])
        self.assertIn("Chat / Ask MRI Assistant", [tab.label for tab in app.tabs])
        self.assertTrue(app.chat_input)
        self.assertTrue(any(toggle.label == "Dark mode" for toggle in app.toggle))
        self.assertTrue(
            any(
                expander.label == "Patient study input (all four modalities)"
                for expander in app.expander
            )
        )
        next(toggle for toggle in app.toggle if toggle.label == "Dark mode").set_value(
            True
        ).run(timeout=20)
        self.assertEqual(app.exception, [])

    def test_registry_records_model_card_3d_checkpoint_contract(self) -> None:
        models = load_model_registry()

        self.assertEqual(
            models["segmentation_2d"]["architecture"], "BRISCAttentionUNet"
        )
        self.assertEqual(
            models["classification_2d"]["architecture"], "BRISCClassifier"
        )
        self.assertEqual(
            models["classification_2d"]["checkpoint"],
            "4_2d_classification/outputs/classification/20261018_214516_seed2026/best.pt",
        )
        self.assertEqual(
            models["classification_2d"]["selection_evidence"]["evaluation_status"],
            "validation_only",
        )
        self.assertEqual(
            models["segmentation_3d"]["status"],
            "user_supplied_checkpoint_strictly_verified",
        )
        self.assertEqual(
            models["segmentation_3d"]["checkpoint"],
            "2_3d_segmentation/outputs/segmentation3d/SegResNet_BraTS2021/model.pt",
        )
        self.assertEqual(models["segmentation_3d"]["output_channels"], 3)
        self.assertEqual(
            models["segmentation_3d"]["architecture"], "SegResNet"
        )
        self.assertEqual(
            models["segmentation_3d"]["input_modalities"],
            ["t1c", "t1n", "t2w", "t2f"],
        )
        self.assertEqual(
            models["segmentation_3d"]["output_order"], ["TC", "WT", "ET"]
        )
        self.assertEqual(
            models["segmentation_3d"]["unavailable_outputs"]["RC"],
            "Not specified by the supplied checkpoint; no RC mask is produced.",
        )
        self.assertEqual(models["segmentation_3d"]["inference"]["roi_size"], [64, 64, 64])
        self.assertEqual(models["segmentation_3d"]["inference"]["sw_batch_size"], 1)
        self.assertEqual(models["segmentation_3d"]["inference"]["overlap"], 0.25)
        self.assertEqual(models["segmentation_3d"]["input_spacing_mm"], [1.0, 1.0, 1.0])

    def test_missing_checkpoints_never_produce_placeholder_predictions(self) -> None:
        models = load_model_registry()
        image_buffer = BytesIO()
        Image.new("L", (4, 4), color=90).save(image_buffer, format="PNG")
        upload = load_mri_upload(image_buffer.getvalue(), "sample.png")
        result = _empty_result("sample", "sample.png", "T2", upload, 0)

        _run_analysis(
            result,
            upload.data,
            upload.metadata,
            str(Path("missing-segmentation.pt")),
            str(Path("missing-classifier.pt")),
            models["segmentation_2d"],
            models["classification_2d"],
        )

        self.assertEqual(result["stages"]["2D segmentation"]["status"], "not configured")
        self.assertEqual(result["stages"]["2D classification"]["status"], "not configured")
        self.assertNotIn("mask", result["segmentation"])
        self.assertNotIn("predicted_class", result["classification"])
        self.assertEqual(
            result["stages"]["3D segmentation"]["status"], "not run"
        )

    def test_report_does_not_invent_outputs_and_exports_pdf(self) -> None:
        report_data = {
            "case_id": "sample",
            "input": {"modality": "T2", "shape": (4, 5), "preprocessing_status": "validated"},
            "stages": {
                "2D segmentation": {
                    "status": "not configured",
                    "reasons": ["Checkpoint is absent."],
                },
                "3D segmentation": {
                    "status": "checkpoint unverified",
                    "reasons": ["Output ontology and preprocessing are unverified."],
                },
            },
        }

        markdown = generate_markdown_report(report_data)
        pdf = generate_pdf_report(report_data)

        self.assertIn("### Segmentation Findings", markdown)
        self.assertIn("Checkpoint is absent.", markdown)
        self.assertNotIn("tumor volume:", markdown.lower())
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_3d_report_records_only_tc_wt_et_and_marks_clinical_status(self) -> None:
        markdown = generate_markdown_report(
            {
                "case_id": "sample",
                "model_3d": "user_supplied_monai_segresnet",
                "segmentation3d_metadata": {
                    "region_order": ["TC", "WT", "ET"],
                    "inference": {"threshold": 0.5},
                    "mask_voxel_count": {"TC": 8, "WT": 12, "ET": 3},
                },
                "stages": {
                    "3D segmentation": {"status": "completed", "reasons": []},
                },
            }
        )

        self.assertIn("Output order: ['TC', 'WT', 'ET']", markdown)
        self.assertIn("Mask voxel counts: {'TC': 8, 'WT': 12, 'ET': 3}", markdown)
        self.assertIn("RC: unavailable", markdown)
        self.assertIn("Clinical performance: unvalidated", markdown)

    def test_corridor_outputs_are_rendered_as_a_markdown_table(self) -> None:
        markdown = generate_markdown_report(
            {
                "risk_mapping": {
                    "corridors": [
                        {
                            "corridor_name": "Candidate A",
                            "path_length_mm": 44.2,
                            "mean_risk": 0.18,
                            "maximum_risk": 0.31,
                            "tract_risk": 0.0,
                            "tumor_coverage": 0.91,
                            "composite_cost": 0.2,
                        }
                    ]
                },
                "stages": {"Surgical corridors": {"status": "completed"}},
            }
        )

        self.assertIn("| Candidate corridor | Length (mm) | Mean risk |", markdown)
        self.assertIn("| Candidate A | 44.2 | 0.18 | 0.31 | 0.0 | 0.91 | 0.2 |", markdown)

    def test_final_pdf_includes_modality_and_segmentation_visualizations(self) -> None:
        image = np.arange(48, dtype=np.float32).reshape(6, 8)
        visualization = BytesIO()
        Image.fromarray(np.full((20, 40, 3), 128, dtype=np.uint8)).save(
            visualization, format="PNG"
        )
        report_data = {
            "case_id": "sample",
            "input": {"modality": "T1 contrast-enhanced", "modalities": ["t2f", "t1n", "t1c", "t2w"]},
            "modalities": ["t2f", "t1n", "t1c", "t2w"],
            "stages": {
                "2D segmentation": {"status": "completed", "reasons": []},
                "3D segmentation": {
                    "status": "checkpoint unverified",
                    "reasons": ["Output ontology and preprocessing are unverified."],
                },
            },
            "segmentation": {
                "status": "completed",
                "foreground_area": "7 pixels",
                "mask": np.ones((6, 8), dtype=np.uint8),
            },
            "classification": {"status": "not configured"},
            "slice_index": 2,
            "slice_image": image,
            "modality_slices": {
                key: image + index for index, key in enumerate(("t2f", "t1n", "t1c", "t2w"))
            },
            "segmentation2d": np.ones((6, 8), dtype=np.uint8),
            "segmentation_visualization": visualization.getvalue(),
        }

        pdf = generate_report(report_data)

        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertIn(b"/Subtype /Image", pdf)

    def test_registry_rejects_parent_relative_checkpoint_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registry.json"
            path.write_text(
                """
                {"models": {
                  "segmentation_2d": {"task": "segmentation", "dimensionality": "2D", "checkpoint": "../outside.pt"},
                  "classification_2d": {"task": "classification", "dimensionality": "2D", "status": "not_configured", "checkpoint": null},
                  "segmentation_3d": {"task": "segmentation", "dimensionality": "3D", "status": "not_configured", "checkpoint": null}
                }}
                """,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "safe project-relative"):
                load_model_registry(path)

    def test_selected_checkpoint_integrity_is_verified_and_overrides_are_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            checkpoint.write_bytes(b"trusted-test-checkpoint")
            model = {
                "resolved_checkpoint": checkpoint.resolve(),
                "checkpoint_integrity": {
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                },
            }
            self.assertEqual(
                verify_checkpoint_integrity(checkpoint, model),
                "verified against checkpoint manifest",
            )
            model["resolved_checkpoint"] = Path(temporary) / "different.pt"
            self.assertIn(
                "override not covered",
                verify_checkpoint_integrity(checkpoint, model),
            )

    def test_rag_context_does_not_claim_a_2d_mask_is_a_3d_volume(self) -> None:
        context = build_patient_context(
            "sample",
            {
                "input": {"modality": "T2", "shape": (4, 5), "slice_index": 1},
                "segmentation": {
                    "status": "completed",
                    "architecture": "BRISCAttentionUNet",
                    "foreground_area": "12 pixels",
                },
                "stages": {},
            },
        )

        self.assertIsNone(context["tumor_volume_ml"])
        self.assertIn("no 3D tumor mask", context["provenance"]["tumor_volume"])
        self.assertEqual(context["segmentation_2d"]["foreground_area"], "12 pixels")

    def test_rag_context_combines_modalities_2d_and_3d_risk_outputs_without_masks(self) -> None:
        modality_inputs = {
            key: {"label": key, "uploaded": True, "shape": [4, 5, 6]}
            for key in ("t2f", "t1n", "t1c", "t2w")
        }
        context = build_patient_context(
            "patient-001",
            {
                "segmentation": {
                    "status": "completed",
                    "architecture": "2D model",
                    "foreground_area": "12 pixels",
                },
                "classification": {
                    "status": "completed",
                    "predicted_class": "glioma",
                    "scores": {"glioma": 2.0},
                    "probabilities": {"glioma": 0.8},
                },
                "stages": {},
            },
            {
                "region_masks": {"WT": np.ones((2, 2, 2), dtype=np.uint8)},
                "output_spacing_mm": [1.0, 1.0, 1.0],
                "modalities": ["t2f", "t1n", "t1c", "t2w"],
                "model_3d": "SegResNet",
                "segmentation3d_metadata": {
                    "region_order": ["TC", "WT", "ET"],
                    "mask_voxel_count": {"WT": 8},
                },
                "risk_mapping": {
                    "risk_map": np.full((2, 2, 2), 0.3, dtype=np.float32),
                    "brain_mask": np.ones((2, 2, 2), dtype=bool),
                    "active_weights": {"anatomical": 1.0},
                    "evidence_status": {"dti": "unavailable"},
                    "qc": {"dti_available": False},
                    "structures": ["example structure"],
                    "corridors": [{"corridor_id": 1, "mean_risk": 0.2}],
                },
                "stages": {
                    "surgical risk mapping": {"status": "completed"},
                },
            },
            modality_inputs,
        )

        self.assertEqual(context["patient_label"], "Patient 001")
        self.assertEqual(
            context["available_modalities"],
            ["t2f", "t1n", "t1c", "t2w", "SynthSeg anatomical labels"],
        )
        self.assertEqual(context["segmentation_2d"]["foreground_area"], "12 pixels")
        self.assertEqual(context["classification"], "glioma")
        self.assertEqual(context["classification_results"]["probabilities"], {"glioma": 0.8})
        self.assertEqual(context["segmentation_3d"]["mask_voxel_count"], {"WT": 8})
        self.assertEqual(context["candidate_corridors"], [{"corridor_id": 1, "mean_risk": 0.2}])
        self.assertIn("2D SEGMENTATION INFERENCE:", context["analysis_results_text"])
        self.assertIn('"foreground_area": "12 pixels"', context["analysis_results_text"])
        self.assertIn("SURGICAL RISK MAPPING INFERENCE:", context["analysis_results_text"])
        self.assertIn('"mean_risk": 0.2', context["analysis_results_text"])
        self.assertNotIn("region_masks", context)
        self.assertNotIn("risk_map", context)
        self.assertEqual(context["provenance"]["raw_mri_sent"], False)
        self.assertEqual(
            context["provenance"]["model_findings_source"],
            "active MRI inference results only",
        )
        self.assertTrue(context["provenance"]["synthetic_history_is_not_an_inference"])

    def test_rag_context_does_not_invent_findings_without_inference_outputs(self) -> None:
        context = build_patient_context("patient-001", {"stages": {}})

        self.assertIn("Fictional demonstration history", context["patient_history"])
        self.assertIsNone(context["tumor_volume_ml"])
        self.assertNotIn("classification", context)
        self.assertNotIn("risk_score", context)
        self.assertNotIn("candidate_corridors", context)

    def test_chat_readiness_reports_missing_assets_and_gemini_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks_path = root / "chunks.json"
            manifest_path = root / "vector_db" / "manifest.json"
            with (
                patch.object(rag_adapter, "RAG_CHUNKS_PATH", chunks_path),
                patch.object(rag_adapter, "RAG_VECTOR_MANIFEST", manifest_path),
                patch.dict("os.environ", {"GEMINI_API_KEY": "replace-with-your-key"}),
                patch(
                    "requests.get",
                    side_effect=requests.ConnectionError("service unavailable"),
                ),
            ):
                status = rag_adapter.rag_readiness("http://127.0.0.1:8000")

        self.assertFalse(status["ready"])
        self.assertFalse(status["chunks_available"])
        self.assertEqual(status["api_status"], "unavailable")
        self.assertEqual(status["gemini_status"], "missing API key")
        self.assertTrue(any("chunks are missing" in item for item in status["missing"]))
        self.assertTrue(any("RAG API is not reachable" in item for item in status["missing"]))
        self.assertTrue(any("Gemini API key is not configured" in item for item in status["missing"]))
        self.assertTrue(any("build_knowledge_base.py" in item for item in status["setup"]))

    def test_chat_readiness_rejects_remote_plain_http_without_request(self) -> None:
        with patch(
            "requests.get",
            side_effect=requests.ConnectionError("RAG API not running"),
        ) as request:
            status = rag_adapter.rag_readiness("http://example.invalid:8000")

        self.assertFalse(
            any("example.invalid" in call.args[0] for call in request.call_args_list)
        )
        self.assertFalse(status["ready"])
        self.assertTrue(any("no health request was sent" in item for item in status["missing"]))


if __name__ == "__main__":
    unittest.main()
