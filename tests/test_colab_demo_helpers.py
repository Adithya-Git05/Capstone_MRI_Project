import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import nibabel as nib
import numpy as np

from notebooks.colab_demo_helpers import make_artifact_zip, validate_nifti_study


class ColabDemoHelperTests(unittest.TestCase):
    def test_rejects_missing_modality(self):
        with self.assertRaisesRegex(ValueError, "Missing required modalities"):
            validate_nifti_study({"T1": "not-read.nii.gz"})

    def test_accepts_aligned_finite_study_and_preserves_affine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.diag([1.2, 1.1, 2.0, 1.0])
            paths = {}
            for i, name in enumerate(("T1", "T1ce", "T2", "FLAIR")):
                path = root / f"{name}.nii.gz"
                values = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10) + i
                nib.save(nib.Nifti1Image(values, affine), path)
                paths[name] = path
            metadata = validate_nifti_study(paths)
            self.assertEqual(metadata["status"], "Completed")
            np.testing.assert_allclose(metadata["affine"], affine)
            self.assertEqual(metadata["modalities"]["FLAIR"]["shape"], [8, 9, 10])

    def test_rejects_affine_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ("T1", "T1ce", "T2", "FLAIR"):
                affine = np.eye(4)
                if name == "T2": affine[0, 3] = 2
                path = root / f"{name}.nii.gz"
                values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
                nib.save(nib.Nifti1Image(values, affine), path)
                paths[name] = path
            with self.assertRaisesRegex(ValueError, "affine"):
                validate_nifti_study(paths)

    def test_zip_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            root.mkdir()
            (root / "status.json").write_text('{"status":"Completed"}')
            archive = make_artifact_zip(root, root / "bundle.zip")
            with zipfile.ZipFile(archive) as zipped:
                self.assertIsNone(zipped.testzip())
                self.assertIn("status.json", zipped.namelist())

    def test_pdf_is_readable_and_zip_contains_it(self):
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "Capstone_Report.pdf"
            SimpleDocTemplate(str(pdf), pagesize=letter).build([
                Paragraph(escape("Stage: Completed <safe text>"), getSampleStyleSheet()["BodyText"])
            ])
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
            archive = make_artifact_zip(root, root / "artifacts.zip")
            with zipfile.ZipFile(archive) as zipped:
                self.assertIn("Capstone_Report.pdf", zipped.namelist())


if __name__ == "__main__":
    unittest.main()
