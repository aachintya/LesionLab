"""Exercise ambiguity decisions and the limits of evidence-based summaries."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from components.paths import DIAGNOSIS_COLS
from utils.result_interpretation import interpret_result


def scores(winner=7, top=.7):
    values = np.full(11, (1 - top) / 10)
    values[winner] = top
    return values


def prediction(values=None, *, clinical=None, dermoscopic=None, metadata=None, mode="fusion"):
    values = scores() if values is None else np.asarray(values)
    branches = {"clinical": scores() if clinical is None else clinical,
                "dermoscopic": scores() if dermoscopic is None else dermoscopic,
                "metadata": scores(1) if metadata is None else metadata}
    return {"mode": mode, "class_code": DIAGNOSIS_COLS[int(np.argmax(values))],
            "scores": [{"class": code, "name": code, "score": float(value)} for code, value in zip(DIAGNOSIS_COLS, values)],
            "branch_scores": branches if mode == "fusion" else {mode: values}}


class InterpretationTests(unittest.TestCase):
    def test_agreeing_images_and_clear_margin_describe_a_model_preference(self):
        report = interpret_result(prediction(), metadata={"age_approx": 0, "skin_tone_class": 0})
        self.assertEqual(report["status"], "model_preference")
        self.assertTrue(report["image_views_agree"])
        self.assertFalse(report["policy"]["calibrated"])
        self.assertEqual(report["provided_metadata_fields"], ["age", "skin tone"])
        # A differing weak metadata branch is not treated as image-view disagreement.
        self.assertEqual(report["modalities"][-1]["class"], "BCC")

    def test_close_scores_and_uniform_scores_are_inconclusive(self):
        values = np.full(11, .02); values[7] = .42; values[1] = .40
        for values in [values, np.ones(11) / 11]:
            report = interpret_result(prediction(values))
            self.assertEqual(report["status"], "inconclusive")
            self.assertIn("close_scores", [row["code"] for row in report["reasons"]])

    def test_exact_margin_boundary_is_stable(self):
        values = np.full(11, .05); values[0] = .30; values[1] = .25
        report = interpret_result(prediction(values, mode="clinical"))
        self.assertEqual(report["status"], "model_preference")
        values[0] -= .000001; values[1] += .000001
        self.assertEqual(interpret_result(prediction(values, mode="clinical"))["status"], "inconclusive")

    def test_view_disagreement_is_visible_even_with_a_large_combined_margin(self):
        report = interpret_result(prediction(dermoscopic=scores(1)))
        self.assertEqual(report["status"], "inconclusive")
        self.assertFalse(report["image_views_agree"])
        self.assertIn("views_disagree", [row["code"] for row in report["reasons"]])

    def test_combined_category_outside_both_image_preferences_is_flagged(self):
        report = interpret_result(prediction(scores(2)))
        self.assertTrue(report["image_views_agree"])
        self.assertIn("fusion_differs", [row["code"] for row in report["reasons"]])

    def test_single_image_and_absent_metadata_are_described_honestly(self):
        report = interpret_result(prediction(mode="clinical"))
        self.assertIsNone(report["image_views_agree"])
        self.assertIn("unavailable", report["agreement_summary"])
        report = interpret_result(prediction(), metadata={"age_approx": float("nan"), "sex": "unknown"})
        self.assertFalse(report["provided_metadata_fields"])
        self.assertIn("No patient metadata", report["metadata_note"])

    def test_invalid_and_inconsistent_predictions_never_get_an_interpretation(self):
        for change in [lambda p: p.update(class_code="BCC"),
                       lambda p: p["scores"].pop(),
                       lambda p: p["scores"][0].update(score=float("nan")),
                       lambda p: p["branch_scores"].pop("dermoscopic"),
                       lambda p: p.update(mode="validator")]:
            record = prediction();change(record)
            with self.assertRaises(ValueError):interpret_result(record)

    def test_summary_preserves_scores_and_does_not_invent_visual_findings(self):
        record = prediction(); before = deepcopy(record)
        report = interpret_result(record)
        self.assertEqual(record["scores"], before["scores"])
        for key in before["branch_scores"]:
            np.testing.assert_array_equal(record["branch_scores"][key], before["branch_scores"][key])
        for word in ["border", "asymmetry", "pigment", "treatment", "benign risk"]:
            self.assertNotIn(word, report["summary"].lower())


if __name__ == "__main__":unittest.main()
