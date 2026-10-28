import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from pipeline.corridor_context import (
    build_structure_overlays,
    load_tract_masks,
    nearby_structures,
)


class TestCorridorContext(unittest.TestCase):
    def setUp(self):
        self.shape = (20, 20, 20)
        self.path = np.zeros(self.shape, dtype=bool)
        self.path[10, 10, 2:12] = True
        self.near = np.zeros(self.shape, dtype=bool)
        self.near[14, 10, 6] = True
        self.far = np.zeros(self.shape, dtype=bool)
        self.far[10, 10, 19] = True
        self.generic = np.zeros(self.shape, dtype=bool)
        self.generic[11, 10, 6] = True

    def test_nearby_structures_filters_distance_and_generic_tracts(self):
        found = nearby_structures(
            self.path,
            (1.0, 1.0, 1.0),
            tract_masks={
                "corpus_callosum": self.near,
                "optic_radiations": self.far,
                "white_matter_tracts": self.generic,
            },
            max_distance_mm=5,
        )
        self.assertEqual([item["name"] for item in found], ["Corpus Callosum"])
        self.assertEqual(found[0]["min_distance_mm"], 4.0)

    def test_nearby_structures_empty_centerline_returns_nothing(self):
        self.assertEqual(
            nearby_structures(np.zeros(self.shape, dtype=bool), (1, 1, 1), tract_masks={"a": self.near}),
            [],
        )

    def test_load_tract_masks_skips_wrong_shape_and_empty_masks(self):
        with tempfile.TemporaryDirectory() as tract_dir:
            affine = np.eye(4)
            for name, data in {
                "good_tract": self.near,
                "empty_tract": np.zeros(self.shape, dtype=bool),
                "wrong_shape": np.ones((5, 5, 5), dtype=bool),
            }.items():
                nib.save(nib.Nifti1Image(data.astype(np.uint8), affine), str(Path(tract_dir) / f"{name}.nii.gz"))
            masks = load_tract_masks(tract_dir, self.shape)
            self.assertEqual(list(masks), ["good_tract"])
            self.assertEqual(load_tract_masks(None, self.shape), {})
            self.assertEqual(load_tract_masks(str(Path(tract_dir) / "missing"), self.shape), {})

    def test_build_structure_overlays_skips_generic_far_and_missing_masks(self):
        def record(name, kind, distance):
            return {
                "structure_name": name,
                "structure_type": kind,
                "tumor_min_distance_mm": distance,
                "label_id": 0,
                "risk_label": "ELEVATED",
                "functional_domain": "Motor",
                "potential_consequence": "Weakness",
            }

        overlays = build_structure_overlays(
            [
                record("Corpus Callosum", "White-Matter Tract", 5.0),
                record("Optic Radiations", "White-Matter Tract", 35.0),
                record("White Matter Tracts", "White-Matter Tract", 1.0),
                record("Missing Tract", "White-Matter Tract", 2.0),
                record("Brain", "Generic tissue", 0.0),
            ],
            None,
            {
                "corpus_callosum": self.near,
                "optic_radiations": self.far,
                "white_matter_tracts": self.generic,
            },
        )
        self.assertEqual([item["name"] for item in overlays], ["Corpus Callosum"])
        self.assertTrue(overlays[0]["mask"].any())


if __name__ == "__main__":
    unittest.main()
