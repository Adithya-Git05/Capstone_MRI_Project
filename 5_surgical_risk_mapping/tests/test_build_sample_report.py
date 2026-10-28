import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from pipeline.build_sample_report import describe_tumor


class TestDescribeTumor(unittest.TestCase):
    def _describe(self, tumor_slices, affine):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mri = np.zeros((40, 30, 20), dtype=np.float32)
            mri[5:35, 5:25, 3:17] = 1
            tumor = np.zeros(mri.shape, dtype=np.uint8)
            tumor[tumor_slices] = 1
            nib.save(nib.Nifti1Image(mri, affine), str(root / "mri.nii.gz"))
            nib.save(nib.Nifti1Image(tumor, affine), str(root / "tumor.nii.gz"))
            return describe_tumor(root / "mri.nii.gz", root / "tumor.nii.gz")

    def test_side_follows_axis_orientation_and_extent_uses_spacing(self):
        slices = (slice(26, 32), slice(10, 20), slice(5, 10))
        right = self._describe(slices, np.diag([2.0, 1.0, 1.0, 1.0]))
        left = self._describe(slices, np.diag([-2.0, 1.0, 1.0, 1.0]))
        self.assertEqual((right["side"], left["side"]), ("right", "left"))
        self.assertFalse(right["crosses_midline"])
        self.assertEqual(right["extent_mm"], {"left_right": 12.0, "front_back": 10.0, "up_down": 5.0})

    def test_tumor_spanning_the_midline_is_flagged(self):
        result = self._describe(
            (slice(15, 25), slice(10, 20), slice(5, 10)), np.eye(4))
        self.assertTrue(result["crosses_midline"])

    def test_empty_or_mismatched_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.float32), np.eye(4)), str(root / "mri.nii.gz"))
            nib.save(nib.Nifti1Image(np.zeros((6, 6, 6), np.uint8), np.eye(4)), str(root / "tumor.nii.gz"))
            with self.assertRaisesRegex(ValueError, "non-empty"):
                describe_tumor(root / "mri.nii.gz", root / "tumor.nii.gz")


if __name__ == "__main__":
    unittest.main()
