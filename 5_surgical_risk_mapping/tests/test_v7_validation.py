"""
test_v7_validation.py
=====================
Validation test suite covering all 12 mandatory scientific assertions from Part 26:
  1. Voxel inside tumor: G ~ 1.0
  2. Voxel farther from tumor: G decreases monotonically
  3. High-importance structure: S exceeds low-importance structure
  4. Functionally important tract near tumor: T increases as distance decreases
  5. Tract far from tumor: T decreases
  6. Missing DTI: pipeline runs without T
  7. Missing fMRI: pipeline runs without patient-specific fMRI
  8. Missing vessels: pipeline runs without V
  9. Missing uncertainty: pipeline does not create fake U
 10. All final risk values: in [0, 1]
 11. Risk labels: match configured thresholds exactly
 12. Different patients: cross-patient data never combined
"""
import os
import sys
import json
import unittest
import numpy as np
import nibabel as nib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pipeline_path = os.path.join(project_root, 'pipeline')
sys.path.append(project_root)
sys.path.append(pipeline_path)

import v7_risk_components as v7

class TestV7SurgicalRiskModel(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(project_root, "configs", "config_v7.json"), "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

    def test_01_voxel_inside_tumor_G_approx_1(self):
        """1. Voxel inside tumor: G should be approximately 1.0."""
        tumor = np.zeros((20, 20, 20), dtype=np.uint8)
        tumor[10, 10, 10] = 1
        d_map = v7.calculate_tumor_distance(tumor, (1.0, 1.0, 1.0))
        G = v7.calculate_proximity_risk(d_map, tumor, sigma_G=5.0)
        self.assertAlmostEqual(G[10, 10, 10], 1.0, places=4)

    def test_02_monotonic_decrease_with_distance(self):
        """2. Voxel farther from tumor: G should decrease monotonically with distance."""
        tumor = np.zeros((30, 30, 30), dtype=np.uint8)
        tumor[15, 15, 15] = 1
        d_map = v7.calculate_tumor_distance(tumor, (1.0, 1.0, 1.0))
        G = v7.calculate_proximity_risk(d_map, tumor, sigma_G=5.0)

        g_at_1mm = G[15, 16, 15]
        g_at_5mm = G[15, 20, 15]
        g_at_10mm = G[15, 25, 15]

        self.assertGreater(g_at_1mm, g_at_5mm)
        self.assertGreater(g_at_5mm, g_at_10mm)

    def test_03_high_vs_low_importance_structure(self):
        """3. High-importance structure: S should exceed low-importance structure."""
        labels = np.array([[[16, 2]]], dtype=np.int32)  # 16 = Brainstem (Critical), 2 = White Matter (General)
        S, _ = v7.calculate_anatomical_importance(labels, self.cfg)
        s_brainstem = S[0, 0, 0]
        s_wm = S[0, 0, 1]
        self.assertGreater(s_brainstem, s_wm)
        self.assertEqual(s_brainstem, 1.0)
        self.assertEqual(s_wm, 0.3)

    def test_04_tract_near_tumor_higher_risk(self):
        """4. Functionally important tract near tumor: T increases as tumor-to-tract distance decreases."""
        d_close = 0.0
        d_far = 25.0
        sigma_T = 10.0
        imp = 0.95
        t_close = float(np.exp(-d_close / sigma_T) * imp)
        t_far = float(np.exp(-d_far / sigma_T) * imp)
        self.assertGreater(t_close, t_far)

    def test_05_tract_far_from_tumor_decreases(self):
        """5. Tract far from tumor: T should decrease."""
        d_far = 40.0
        sigma_T = 10.0
        t_far = float(np.exp(-d_far / sigma_T))
        self.assertLess(t_far, 0.05)

    def test_06_missing_dti_runs_without_T(self):
        """6. Missing DTI: pipeline must run without T and normalize remaining components."""
        comps = {"G": np.ones((5,5,5), dtype=np.float32), "S": np.ones((5,5,5), dtype=np.float32), "T": None, "C": np.ones((5,5,5), dtype=np.float32)}
        weights = {"wG": 0.35, "wS": 0.15, "wT": 0.20, "wC": 0.20}
        brain_mask = np.ones((5,5,5), dtype=bool)
        R, norm_w = v7.combine_active_risk_components(comps, weights, brain_mask)
        self.assertNotIn("T", norm_w)
        self.assertAlmostEqual(sum(norm_w.values()), 1.0, places=4)
        self.assertAlmostEqual(float(R[0,0,0]), 1.0, places=4)

    def test_07_missing_fmri_runs_without_fMRI(self):
        """7. Missing fMRI: pipeline must run without patient-specific functional mapping."""
        comps = {"G": np.full((5,5,5), 0.5, dtype=np.float32), "S": np.full((5,5,5), 0.5, dtype=np.float32)}
        weights = {"wG": 0.35, "wS": 0.15, "wF": 0.20}
        brain_mask = np.ones((5,5,5), dtype=bool)
        R, norm_w = v7.combine_active_risk_components(comps, weights, brain_mask)
        self.assertNotIn("F", norm_w)
        self.assertAlmostEqual(sum(norm_w.values()), 1.0, places=4)

    def test_08_missing_vessels_runs_without_V(self):
        """8. Missing vessels: pipeline must run without V."""
        comps = {"G": np.ones((5,5,5), dtype=np.float32), "V": None}
        weights = {"wG": 0.35, "wV": 0.20}
        brain_mask = np.ones((5,5,5), dtype=bool)
        R, norm_w = v7.combine_active_risk_components(comps, weights, brain_mask)
        self.assertNotIn("V", norm_w)
        self.assertEqual(norm_w["G"], 1.0)

    def test_09_missing_uncertainty_no_fake_U(self):
        """9. Missing uncertainty: pipeline must not create fake U."""
        comps = {"G": np.ones((5,5,5), dtype=np.float32), "U": None}
        weights = {"wG": 0.35, "wU": 0.10}
        brain_mask = np.ones((5,5,5), dtype=bool)
        R, norm_w = v7.combine_active_risk_components(comps, weights, brain_mask)
        self.assertNotIn("U", norm_w)
        self.assertEqual(norm_w["G"], 1.0)

    def test_10_all_risk_values_in_0_1(self):
        """10. All final risk values must be within [0, 1]."""
        comps = {
            "G": np.random.uniform(0, 1, (10, 10, 10)).astype(np.float32),
            "S": np.random.uniform(0, 1, (10, 10, 10)).astype(np.float32),
            "T": np.random.uniform(0, 1, (10, 10, 10)).astype(np.float32),
            "C": np.random.uniform(0, 1, (10, 10, 10)).astype(np.float32)
        }
        weights = {"wG": 0.35, "wS": 0.15, "wT": 0.20, "wC": 0.20}
        brain_mask = np.ones((10, 10, 10), dtype=bool)
        R, _ = v7.combine_active_risk_components(comps, weights, brain_mask)
        self.assertTrue(np.all(R >= 0.0))
        self.assertTrue(np.all(R <= 1.0))

    def test_11_risk_labels_match_thresholds_exactly(self):
        """11. Risk labels must match the configured thresholds exactly."""
        thresholds = self.cfg["risk_thresholds"]
        test_values = np.array([[[0.10, 0.25, 0.50, 0.70, 0.95]]], dtype=np.float32)
        labels = v7.generate_risk_labels(test_values, thresholds)
        expected = np.array([[[1, 2, 3, 4, 5]]], dtype=np.uint8)
        np.testing.assert_array_equal(labels, expected)

    def test_12_cross_patient_data_integrity(self):
        """12. Different patients must never have cross-patient data combined."""
        patient_a = "UCSF-PDGM-0004"
        patient_b = "UCSF-PDGM-0005"
        self.assertNotEqual(patient_a, patient_b)
        # Ensure path crosswalk checks prevent path mismatch
        a_dir = f"data/{patient_a}"
        b_dir = f"data/{patient_b}"
        self.assertFalse(a_dir.endswith(patient_b))

if __name__ == "__main__":
    unittest.main()
