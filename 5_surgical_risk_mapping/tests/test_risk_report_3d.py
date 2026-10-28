import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from pipeline.risk_report_3d import generate_risk_3d_report


class TestRiskReport3D(unittest.TestCase):
    def _save_volume(self, path: Path, data: np.ndarray, affine: np.ndarray) -> None:
        nib.save(nib.Nifti1Image(data, affine), str(path))

    def test_builds_report_without_any_grad_cam_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shape = (12, 14, 10)
            affine = np.diag([1.2, 1.3, 1.4, 1.0])
            grid = np.indices(shape)
            inside = (grid[0] - 6) ** 2 + (grid[1] - 7) ** 2 + (grid[2] - 5) ** 2 < 25
            mri = np.where(inside, grid[0] + 100, 0).astype(np.float32)
            tumor = np.zeros(shape, dtype=np.uint8)
            tumor[5:8, 6:9, 4:7] = 1
            risk = np.zeros(shape, dtype=np.float32)
            risk[3:10, 4:11, 2:8] = 0.65
            for name, data in (("mri.nii.gz", mri), ("tumor.nii.gz", tumor), ("risk.nii.gz", risk)):
                self._save_volume(root / name, data, affine)

            result = generate_risk_3d_report(
                str(root / "mri.nii.gz"),
                str(root / "tumor.nii.gz"),
                str(root / "risk.nii.gz"),
                str(root / "base.html"),
                max_points=1000,
            )

            page = (root / "base.html").read_text(encoding="utf-8")
            self.assertTrue((root / "base.json").is_file())
            self.assertNotIn("grad-cam", page.lower())
            self.assertNotIn("gradcam", result)
            self.assertNotIn("gradcam_status", result)
            self.assertIn("showSendToCloud", page)
            self.assertEqual(result["coordinate_space"], "NIfTI affine transformed to RAS+ millimeters")

    def test_rejects_misaligned_risk_volume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shape = (8, 8, 8)
            affine = np.eye(4)
            mri = np.ones(shape, dtype=np.float32)
            mri[0, 0, 0] = 2
            tumor = np.zeros(shape, dtype=np.uint8)
            tumor[2:4, 2:4, 2:4] = 1
            self._save_volume(root / "mri.nii.gz", mri, affine)
            self._save_volume(root / "tumor.nii.gz", tumor, affine)
            changed_affine = affine.copy()
            changed_affine[0, 3] = 3
            self._save_volume(root / "risk.nii.gz", np.zeros(shape, dtype=np.float32), changed_affine)

            with self.assertRaisesRegex(ValueError, "affine does not match"):
                generate_risk_3d_report(
                    str(root / "mri.nii.gz"),
                    str(root / "tumor.nii.gz"),
                    str(root / "risk.nii.gz"),
                    str(root / "report.html"),
                )

    def _case_files(self, root: Path):
        shape = (16, 16, 16)
        affine = np.eye(4)
        grid = np.indices(shape)
        inside = (grid[0] - 8) ** 2 + (grid[1] - 8) ** 2 + (grid[2] - 8) ** 2 < 49
        mri = np.where(inside, 100 + grid[0], 0).astype(np.float32)
        tumor = np.zeros(shape, dtype=np.uint8)
        tumor[7:10, 7:10, 7:10] = 1
        risk = np.where(inside, 0.4, 0).astype(np.float32)
        corridor_volume = np.zeros(shape, dtype=np.uint8)
        corridor_volume[8, 8, 1:7] = 1
        for name, data in (
            ("mri.nii.gz", mri),
            ("tumor.nii.gz", tumor),
            ("risk.nii.gz", risk),
            ("corridors.nii.gz", corridor_volume),
        ):
            self._save_volume(root / name, data, affine)
        corridors = {
            "target": {"world_mm": [8.0, 8.0, 8.0]},
            "corridors": [{
                "corridor_id": 1,
                "name": "C1 <i>Test</i> approach",
                "length_mm": 7.0,
                "mean_risk": 0.4,
                "max_risk": 0.5,
                "pareto_efficient": True,
                "path_world_mm": [[8, 8, z] for z in range(1, 8)],
                "nearby_structures": [
                    {"name": "Corpus Callosum", "kind": "White-matter tract", "min_distance_mm": 4.0},
                ],
            }],
        }
        (root / "corridors.json").write_text(json.dumps(corridors), encoding="utf-8")
        return shape, corridors

    def test_renders_corridors_structures_and_slice_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shape, _ = self._case_files(root)
            structure = np.zeros(shape, dtype=bool)
            structure[10:13, 7:10, 7:10] = True
            result = generate_risk_3d_report(
                str(root / "mri.nii.gz"),
                str(root / "tumor.nii.gz"),
                str(root / "risk.nii.gz"),
                str(root / "report.html"),
                max_points=2000,
                corridors_path=str(root / "corridors.json"),
                corridor_volume_path=str(root / "corridors.nii.gz"),
                structure_overlays=[{
                    "name": "Corpus Callosum",
                    "kind": "White-Matter Tract",
                    "mask": structure,
                    "distance_mm": 0.0,
                    "risk_label": "ELEVATED",
                    "domain": "Connectivity",
                    "consequence": "Example",
                }],
            )
            page = (root / "report.html").read_text(encoding="utf-8")
            self.assertEqual(result["corridor_count"], 1)
            self.assertIn("Candidate approach corridors", page)
            self.assertIn("Named structures near the tumor", page)
            self.assertIn("C1 &lt;i&gt;Test&lt;/i&gt; approach", page)
            self.assertNotIn("C1 <i>Test</i>", page)
            self.assertNotIn("__SUMMARY__", page)
            self.assertNotIn("__PLOT__", page)
            self.assertIn("Corpus Callosum", page)

    def test_rejects_bad_corridor_file_and_overlay_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shape, corridors = self._case_files(root)
            args = (
                str(root / "mri.nii.gz"),
                str(root / "tumor.nii.gz"),
                str(root / "risk.nii.gz"),
                str(root / "report.html"),
            )
            bad = {"corridors": [dict(corridors["corridors"][0], path_world_mm=[[0, 0, float("nan")]] * 2)]}
            (root / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite"):
                generate_risk_3d_report(*args, corridors_path=str(root / "bad.json"))
            (root / "worse.json").write_text(json.dumps({"corridors": "none"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "'corridors' list"):
                generate_risk_3d_report(*args, corridors_path=str(root / "worse.json"))
            with self.assertRaisesRegex(ValueError, "does not match the MRI shape"):
                generate_risk_3d_report(
                    *args,
                    structure_overlays=[{"name": "Bad", "mask": np.ones((3, 3, 3), dtype=bool)}],
                )


if __name__ == "__main__":
    unittest.main()
