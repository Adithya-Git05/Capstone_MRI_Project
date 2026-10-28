from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from integrated_pipeline.modules.model_registry import load_model_registry
from src.data.brats import REGION_LABELS, REGION_ORDER, build_region_masks
from src.data.brats import BraTSCase, load_input_volume
from src.segmentation3d.metrics import region_metrics
from src.segmentation3d.monai_engine import (
    ONTOLOGY_ID,
    _ontology_metadata,
    _regions,
    _require_new_or_empty_output_dir,
    _resolve_inference_device,
    _save_evaluation,
    _save_prediction_nifti,
    _summarize,
    _validate_checkpoint_ontology,
    predict_registered_3d_case,
    predict_segresnet_case,
    verify_segresnet_checkpoint,
    verify_segresnet_checkpoint,
)
from src.segmentation3d.nnunet_adapter import (
    DATASET_NAME,
    _assert_generated_dataset_regions,
    _nnunet_region_labels,
    _nnunet_region_predictions,
    _reorient_nnunet_region_predictions,
)


class Segmentation3DOntologyTests(unittest.TestCase):
    def test_inference_device_auto_uses_cuda_when_available_else_cpu(self) -> None:
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(str(_resolve_inference_device("auto")), expected)
        self.assertEqual(str(_resolve_inference_device("cpu")), "cpu")

    def test_inference_device_rejects_unavailable_cuda_and_other_backends(self) -> None:
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                _resolve_inference_device("cuda")
        with self.assertRaisesRegex(ValueError, "only CPU or CUDA"):
            _resolve_inference_device("mps")

    def test_model_card_input_order_normalization_and_native_affine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case_id = "synthetic-case"
            case_dir = Path(temporary)
            affine = np.diag([-1.5, 2.0, 3.0, 1.0])
            affine[0, 3] = 20.0
            base = np.arange(1, 61, dtype=np.float32).reshape(3, 4, 5)
            raw_by_modality = {
                modality: np.roll(base, shift=index, axis=0)
                for index, modality in enumerate(("t1n", "t1c", "t2w", "t2f"))
            }
            for modality, data in raw_by_modality.items():
                nib.save(
                    nib.Nifti1Image(data, affine),
                    case_dir / f"{case_id}-{modality}.nii.gz",
                )
            case = BraTSCase(
                case_id=case_id,
                patient_group=case_id,
                directory=case_dir,
            )

            volume = load_input_volume(
                case,
                modalities=("t1n", "t1c", "t2w", "t2f"),
                orientation=None,
            )

        self.assertEqual(tuple(volume["image"].shape), (4, 3, 4, 5))
        np.testing.assert_allclose(volume["affine"], affine)
        for channel, modality in enumerate(("t1n", "t1c", "t2w", "t2f")):
            data = raw_by_modality[modality]
            expected = (data - data.mean()) / data.std()
            np.testing.assert_allclose(volume["image"][channel].numpy(), expected)

    def test_approved_raw_label_mapping_and_independent_overlaps(self) -> None:
        self.assertEqual(REGION_ORDER, ("WT", "TC", "ET", "RC"))
        self.assertEqual(
            REGION_LABELS,
            {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "RC": (4,)},
        )
        raw = np.asarray([0, 1, 2, 3, 4], dtype=np.uint8)
        masks = build_region_masks(raw, {"RC": [4], "ET": [3], "TC": [1, 3], "WT": [1, 2, 3]})
        expected = np.asarray(
            [
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(masks, expected)
        self.assertEqual(tuple(masks[:, 3].tolist()), (1.0, 1.0, 1.0, 0.0))

    def test_config_requires_explicit_approved_channel_order_and_mapping(self) -> None:
        self.assertEqual(
            _regions(
                {
                    "label_scheme": ONTOLOGY_ID,
                    "region_order": ["WT", "TC", "ET", "RC"],
                    "region_labels": {
                        "WT": [1, 2, 3],
                        "TC": [1, 3],
                        "ET": [3],
                        "RC": [4],
                    },
                }
            ),
            {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3], "RC": [4]},
        )
        with self.assertRaisesRegex(ValueError, "Legacy L1-L4"):
            _regions({"label_scheme": "nnunet_regions"})
        with self.assertRaisesRegex(ValueError, "region_order"):
            _regions({"label_scheme": ONTOLOGY_ID, "region_order": ["RC", "ET", "TC", "WT"]})

    def test_four_channel_segmentation_model_output(self) -> None:
        from src.segmentation3d.models import build_monai_model

        model = build_monai_model(
            "segresnet", patch_size=(32, 32, 32), in_channels=4, out_channels=4, feature_size=8
        ).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 4, 32, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 4, 32, 32, 32))

    def test_segresnet_inference_preserves_tc_wt_et_order_and_native_geometry(self) -> None:
        from src.segmentation3d.models import build_monai_model

        model_spec = dict(load_model_registry()["segmentation_3d"])
        model_spec["inference"] = {
            **model_spec["inference"],
            "roi_size": [32, 32, 32],
            "overlap": 0.25,
        }
        case_id = "segresnet-smoke"
        affine = np.diag([1.0, 1.0, 1.0, 1.0])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / case_id
            case_dir.mkdir()
            checkpoint = root / "model.pt"
            model = build_monai_model(
                "segresnet",
                patch_size=(32, 32, 32),
                in_channels=4,
                out_channels=3,
                feature_size=16,
            )
            torch.save(model.state_dict(), checkpoint)
            del model

            generator = np.random.default_rng(42)
            for modality in ("t1c", "t1n", "t2w", "t2f"):
                image = generator.normal(size=(32, 32, 32)).astype(np.float32)
                nib.save(
                    nib.Nifti1Image(image, affine),
                    case_dir / f"{case_id}-{modality}.nii.gz",
                )

            compatibility = verify_segresnet_checkpoint(checkpoint, model_spec)
            self.assertEqual(compatibility["status"], "compatible")
            self.assertEqual(compatibility["state_dict_tensor_count"], 83)

            output_dir = root / "prediction"
            predict_segresnet_case(
                checkpoint, case_dir, output_dir, model_spec, device="cpu"
            )
            metadata = json.loads(
                (output_dir / "prediction_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["region_order"], ["TC", "WT", "ET"])
            self.assertIn("RC", metadata["unavailable_regions"])
            self.assertEqual(metadata["device"], "cpu")
            self.assertEqual(metadata["output_shape"], [32, 32, 32])
            self.assertTrue(
                (output_dir / "segmentation_visualization.png")
                .read_bytes()
                .startswith(b"\x89PNG\r\n\x1a\n")
            )
            for region in ("TC", "WT", "ET"):
                output = nib.load(str(output_dir / f"{case_id}_{region}.nii.gz"))
                self.assertEqual(output.shape, (32, 32, 32))
                np.testing.assert_allclose(output.affine, affine, atol=1e-4)
                self.assertTrue(
                    np.isin(output.get_fdata(dtype=np.float32), (0, 1)).all()
                )
                summary = metadata["probability_summary"][region]
                self.assertGreaterEqual(summary["minimum"], 0.0)
                self.assertLessEqual(summary["maximum"], 1.0)

    def test_segresnet_rejects_non_isotropic_input_without_resampling(self) -> None:
        model_spec = dict(load_model_registry()["segmentation_3d"])
        case_id = "wrong-spacing"
        affine = np.diag([2.0, 2.0, 2.0, 1.0])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / case_id
            case_dir.mkdir()
            checkpoint = root / "placeholder.pt"
            checkpoint.write_bytes(b"not loaded before spacing validation")
            for index, modality in enumerate(("t1c", "t1n", "t2w", "t2f")):
                data = np.arange(8**3, dtype=np.float32).reshape(8, 8, 8) + index
                nib.save(
                    nib.Nifti1Image(data, affine),
                    case_dir / f"{case_id}-{modality}.nii.gz",
                )

            with self.assertRaisesRegex(ValueError, "requires 1 mm isotropic input"):
                predict_segresnet_case(
                    checkpoint,
                    case_dir,
                    root / "prediction",
                    model_spec,
                    device="cpu",
                )

    def test_region_metrics_follow_canonical_channels_and_empty_behavior(self) -> None:
        target = np.zeros((4, 2, 2, 2), dtype=bool)
        target[0, 0, 0, 0] = True
        target[1, 0, 0, 0] = True
        target[2, 0, 0, 0] = True
        prediction = target.copy()
        prediction[1, 1, 1, 1] = True
        result = region_metrics(
            prediction, target, list(REGION_ORDER), (1.0, 1.0, 1.0), compute_hd95=False
        )
        self.assertEqual(list(result), list(REGION_ORDER))
        self.assertEqual(result["WT"]["dice"], 1.0)
        self.assertEqual(result["TC"]["dice"], 2 / 3)
        self.assertEqual(result["RC"]["dice"], 1.0)
        self.assertEqual(result["RC"]["iou"], 1.0)
        self.assertIsNone(result["RC"]["hd95_mm"])
        with self.assertRaisesRegex(ValueError, "channel order"):
            region_metrics(prediction, target, ["L1", "L2", "L3", "L4"], (1.0, 1.0, 1.0))

    def test_metrics_and_ontology_metadata_name_regions(self) -> None:
        rows = [
            {"case_id": "case", "region": name, "dice": 1.0, "iou": 1.0, "hd95_mm": None}
            for name in REGION_ORDER
        ]
        report = _summarize(rows, "synthetic", None, {name: list(values) for name, values in REGION_LABELS.items()})
        self.assertEqual(report["region_order"], ["WT", "TC", "ET", "RC"])
        self.assertEqual(list(report["per_region"]), ["WT", "TC", "ET", "RC"])
        self.assertEqual(report["region_definitions"]["RC"], [4])
        self.assertEqual(_ontology_metadata()["channel_order"], ["WT", "TC", "ET", "RC"])
        with tempfile.TemporaryDirectory() as temp_dir:
            _save_evaluation(Path(temp_dir), rows, report, [])
            saved = json.loads((Path(temp_dir) / "metrics.json").read_text())
            self.assertEqual(list(saved["per_region"]), ["WT", "TC", "ET", "RC"])
            self.assertEqual(saved["ontology"]["channel_order"], ["WT", "TC", "ET", "RC"])

    def test_legacy_or_mismatched_checkpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no verified.*ontology"):
            _validate_checkpoint_ontology({"model_state": {}})
        config = {
            "label_scheme": ONTOLOGY_ID,
            "region_order": ["WT", "TC", "ET", "RC"],
            "region_labels": {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3], "RC": [4]},
        }
        checkpoint = {"ontology": _ontology_metadata(), "config": config}
        _validate_checkpoint_ontology(checkpoint)
        checkpoint["ontology"] = {"ontology_id": ONTOLOGY_ID, "channel_order": ["L1", "L2", "L3", "L4"]}
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_checkpoint_ontology(checkpoint)

    def test_nnunet_regions_and_probabilities_preserve_overlap(self) -> None:
        self.assertEqual(
            _nnunet_region_labels(),
            {"background": 0, "WT": [1, 2, 3], "TC": [1, 3], "ET": [3], "RC": [4]},
        )
        probabilities = np.zeros((4, 1, 1, 1), dtype=np.float32)
        probabilities[:3, 0, 0, 0] = 0.8
        masks = _nnunet_region_predictions(probabilities)
        self.assertEqual(tuple(masks[:, 0, 0, 0]), (True, True, True, False))
        reoriented = _reorient_nnunet_region_predictions(probabilities, np.eye(4))
        self.assertEqual(tuple(reoriented[:, 0, 0, 0]), (True, True, True, False))
        with self.assertRaisesRegex(ValueError, r"shape \[4"):
            _nnunet_region_predictions(np.zeros((3, 1, 1, 1), dtype=np.float32))

    def test_nnunet_dataset_declares_and_enforces_channel_order(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / DATASET_NAME
            dataset_dir.mkdir()
            dataset_config = {
                "labels": _nnunet_region_labels(),
                "region_order": ["WT", "TC", "ET", "RC"],
                "ontology_id": ONTOLOGY_ID,
                "regions_class_order": [1, 2, 3, 4],
            }
            path = dataset_dir / "dataset.json"
            path.write_text(json.dumps(dataset_config))
            _assert_generated_dataset_regions(dataset_dir)
            dataset_config["labels"] = {
                "background": 0,
                "RC": [4],
                "ET": [3],
                "TC": [1, 3],
                "WT": [1, 2, 3],
            }
            path.write_text(json.dumps(dataset_config))
            with self.assertRaisesRegex(ValueError, "channel order"):
                _assert_generated_dataset_regions(dataset_dir)

    def test_existing_output_directory_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "existing"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):
                _require_new_or_empty_output_dir(output_dir)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_prediction_filenames_encode_canonical_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "masks"
            masks = np.zeros((4, 2, 2, 2), dtype=bool)
            _save_prediction_nifti(
                output_dir,
                "case-001",
                masks,
                list(REGION_ORDER),
                np.eye(4),
            )
            self.assertEqual(
                sorted(path.name for path in output_dir.glob("*.nii.gz")),
                [
                    "case-001_ET.nii.gz",
                    "case-001_RC.nii.gz",
                    "case-001_TC.nii.gz",
                    "case-001_WT.nii.gz",
                ],
            )


@unittest.skipUnless(
    os.environ.get("MRI_3D_SMOKE_CASE_DIR"),
    "Set MRI_3D_SMOKE_CASE_DIR to a complete four-modality NIfTI study.",
)
class SegResNetInferenceSmokeTests(unittest.TestCase):
    def test_real_checkpoint_inference_preserves_masks_and_visualization(self) -> None:
        case_dir = Path(os.environ["MRI_3D_SMOKE_CASE_DIR"]).expanduser().resolve()
        case_id = os.environ.get("MRI_3D_SMOKE_CASE_ID", case_dir.name)
        model_spec = load_model_registry()["segmentation_3d"]
        checkpoint = Path(model_spec["resolved_checkpoint"])
        modalities = tuple(model_spec["input_modalities"])
        case = BraTSCase(
            case_id=case_id,
            patient_group=case_id.rsplit("-", 1)[0],
            directory=case_dir,
        )

        for modality in modalities:
            self.assertTrue(
                (case_dir / f"{case_id}-{modality}.nii.gz").is_file(),
                f"Missing required {modality} NIfTI input.",
            )
        compatibility = verify_segresnet_checkpoint(checkpoint, model_spec)
        self.assertEqual(compatibility["status"], "compatible")
        source = load_input_volume(case, modalities=modalities, orientation=None)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "prediction"
            predict_registered_3d_case(
                checkpoint, case_dir, output_dir, model_spec
            )
            metadata = json.loads(
                (output_dir / "prediction_metadata.json").read_text(encoding="utf-8")
            )

            self.assertEqual(metadata["modalities"], ["t1c", "t1n", "t2w", "t2f"])
            self.assertEqual(metadata["region_order"], ["TC", "WT", "ET"])
            self.assertIn("RC", metadata["unavailable_regions"])
            visualization = output_dir / metadata["visualization"]
            self.assertTrue(visualization.is_file())
            self.assertTrue(visualization.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

            for region in ("TC", "WT", "ET"):
                image = nib.load(str(output_dir / f"{case_id}_{region}.nii.gz"))
                self.assertEqual(tuple(image.shape), tuple(source["image"].shape[1:]))
                np.testing.assert_allclose(image.affine, source["affine"], atol=1e-4)
                mask = image.get_fdata(dtype=np.float32)
                self.assertTrue(np.isfinite(mask).all())
                self.assertTrue(np.isin(mask, (0, 1)).all())
                summary = metadata["probability_summary"][region]
                self.assertTrue(
                    all(
                        np.isfinite(summary[key])
                        for key in ("minimum", "maximum", "mean")
                    )
                )
                self.assertGreaterEqual(summary["minimum"], 0.0)
                self.assertLessEqual(summary["maximum"], 1.0)


if __name__ == "__main__":
    unittest.main()
