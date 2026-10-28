import json
import tempfile
import unittest
from pathlib import Path

from pipeline.clinical_case_report import generate_clinical_case_report


class TestClinicalCaseReport(unittest.TestCase):
    def test_completed_report_uses_mask_measurement_and_safe_research_language(self):
        with tempfile.TemporaryDirectory() as output_dir:
            report_path = Path(output_dir) / "summary.html"
            generate_clinical_case_report(
                {
                    "case_id": "case-001",
                    "patient_details": {
                        "age": 52,
                        "sex": "Female",
                        "history": "Supplied history",
                    },
                    "missing_inputs": [],
                },
                {
                    "tumor_volume_mm3": 1234.5,
                    "top_risk_structures": [{
                        "structure_name": "<script>alert(1)</script>",
                        "risk_label": "ELEVATED",
                        "tumor_min_distance_mm": 4.2,
                    }],
                },
                report_path,
            )
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("1.23 cm³", report)
            self.assertIn("52 years", report)
            self.assertIn("&lt;script&gt;", report)
            self.assertNotIn("<script>", report)
            self.assertIn("not clinically validated", report)
            self.assertIn("not a diagnosis", report.lower())
            self.assertNotIn("checkpoint", report.lower())
            self.assertNotIn("classifier", report.lower())
            self.assertTrue(report_path.with_suffix(".json").is_file())

    def test_corridor_section_is_plain_language_unranked_and_escaped(self):
        corridors = [{
            "name": "C2 <b>Superior</b>",
            "length_mm": 35.2,
            "mean_risk": 0.335,
            "max_risk": 0.6,
            "nearby_structures": [
                {"name": "Corticospinal Tract", "min_distance_mm": 0.0},
                {"name": "Corpus Callosum", "min_distance_mm": 9.0},
            ],
        }]
        with tempfile.TemporaryDirectory() as output_dir:
            report_path = Path(output_dir) / "summary.html"
            generate_clinical_case_report(
                {"case_id": "c", "patient_details": {}, "missing_inputs": []},
                {"tumor_volume_mm3": 1000.0, "top_risk_structures": []},
                report_path,
                corridors,
            )
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Candidate approach routes", report)
            self.assertIn("not a surgical plan", report)
            self.assertIn("not ranked", report)
            self.assertIn("the path passes through it", report)
            self.assertIn("9 mm away", report)
            self.assertIn("C2 &lt;b&gt;Superior", report)
            self.assertNotIn("<b>Superior", report)
            self.assertEqual(
                json.loads(report_path.with_suffix(".json").read_text(encoding="utf-8"))[
                    "candidate_corridors"][0]["name"],
                "C2 <b>Superior</b>",
            )

    def test_report_embeds_viewer_and_sample_note_safely(self):
        viewer = '<html><body><script>var a="x";</script><p>3D "viewer"</p></body></html>'
        with tempfile.TemporaryDirectory() as output_dir:
            report_path = Path(output_dir) / "summary.html"
            generate_clinical_case_report(
                {"case_id": "c", "patient_details": {}, "missing_inputs": []},
                {"tumor_volume_mm3": 1000.0, "top_risk_structures": [{
                    "structure_name": "White Matter Tracts", "risk_label": "ELEVATED",
                    "tumor_min_distance_mm": 0.0}, {
                    "structure_name": "Corticospinal Tract", "functional_domain": "Motor",
                    "risk_label": "HIGH", "tumor_min_distance_mm": 0.0}]},
                report_path,
                tumor_details={
                    "side": "left", "crosses_midline": False,
                    "extent_mm": {"left_right": 40.0, "front_back": 50.0, "up_down": 30.0},
                },
                viewer_html=viewer,
                sample_note="Example <b>only</b>",
            )
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("<iframe", report)
            self.assertIn("srcdoc=", report)
            self.assertNotIn("<script>var a", report)
            self.assertIn("&lt;script&gt;var a=&quot;x&quot;", report)
            self.assertIn("Example &lt;b&gt;only&lt;/b&gt;", report)
            self.assertIn("Mainly in the left half of the brain", report)
            self.assertIn("40 mm left-right", report)
            self.assertIn("Overlaps the annotated tumor", report)
            self.assertNotIn("White Matter Tracts", report)
            self.assertIn("Not supplied", report)

    def test_classification_is_shown_with_source_or_marked_not_available(self):
        metadata = {"case_id": "c", "patient_details": {}, "missing_inputs": []}
        summary = {"tumor_volume_mm3": 1000.0, "top_risk_structures": []}
        with tempfile.TemporaryDirectory() as output_dir:
            supplied = Path(output_dir) / "supplied.html"
            record = generate_clinical_case_report(
                metadata, summary, supplied,
                classification={"label": "High-grade <glioma>", "source": "Supplied by team"},
            )
            report = supplied.read_text(encoding="utf-8")
            self.assertIn("Tumor classification", report)
            self.assertIn("High-grade &lt;glioma&gt;", report)
            self.assertIn("Supplied by team", report)
            self.assertIn("not determined or verified by this software", report)
            self.assertNotIn("Not available.", report)
            self.assertEqual(record["tumor_classification"]["label"], "High-grade <glioma>")
            self.assertIn("supplied with the case", record["provenance"]["tumor_classification_source"])

            missing = Path(output_dir) / "missing.html"
            record = generate_clinical_case_report(metadata, summary, missing)
            report = missing.read_text(encoding="utf-8")
            self.assertIn("<strong>Not available.</strong>", report)
            self.assertIsNone(record["tumor_classification"])
            self.assertEqual(record["provenance"]["tumor_classification_source"], "not available")

            for bad in ({"label": "HGG"}, {"source": "pathology"}, {"label": "x" * 301, "source": "s"}):
                with self.assertRaises(ValueError):
                    generate_clinical_case_report(
                        metadata, summary, Path(output_dir) / "bad.html", classification=bad)

    def test_report_without_corridors_has_no_route_section(self):
        with tempfile.TemporaryDirectory() as output_dir:
            report_path = Path(output_dir) / "summary.html"
            generate_clinical_case_report(
                {"case_id": "c", "patient_details": {}, "missing_inputs": []},
                {"tumor_volume_mm3": 1000.0, "top_risk_structures": []},
                report_path,
            )
            self.assertNotIn("Candidate approach routes", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
