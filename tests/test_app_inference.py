"""Check rejection ordering, fusion inputs, and safe checkpoint import for the GUI."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.app_inference import LesionPredictor
from utils.input_validation import InputRejected, ValidationResult
from utils.model_artifacts import REQUIRED_FILES, install_artifacts


class AppInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for name in REQUIRED_FILES:
            (self.directory / name).write_bytes(b"test checkpoint")
        with patch("utils.app_inference.InputValidator") as validator:
            validator.return_value.require.return_value = ValidationResult("accepted", "Supported")
            self.predictor = LesionPredictor(self.directory, device="cpu")

    def test_reject_either_input_before_any_disease_preprocessing(self):
        self.predictor.validator.require.side_effect = [ValidationResult("accepted", "Supported"),
            InputRejected(ValidationResult("unsupported_image", "Upload a lesion image"), "bottle")]
        with patch.object(self.predictor, "_image_probs") as inference:
            with self.assertRaises(InputRejected):
                self.predictor.analyze({"clinical": "lesion", "dermoscopic": "bottle"}, mode="fusion", include_heatmaps=True)
            inference.assert_not_called()
        self.assertFalse(self.predictor.models)

    def test_validator_only_never_loads_a_disease_model(self):
        with patch.object(self.predictor, "_image_probs") as inference:
            result = self.predictor.analyze({"image": "lesion"}, mode="validator")
        self.assertEqual(result["scores"], [])
        self.assertNotIn("class_name", result)
        inference.assert_not_called()

    def test_fusion_uses_clinical_dermoscopic_metadata_in_training_order(self):
        clinical = np.arange(1, 12, dtype=float); clinical /= clinical.sum()
        dermoscopic = clinical[::-1].copy()
        metadata = np.ones(11) / 11
        fusion = Mock()
        fusion.predict_proba.return_value = clinical[None, :]
        self.predictor.models["fusion"] = fusion
        with patch.object(self.predictor, "_image_probs", side_effect=[clinical, dermoscopic]), \
                patch.object(self.predictor, "_metadata_probs", return_value=metadata):
            result = self.predictor.analyze({"clinical": "c", "dermoscopic": "d"}, mode="fusion")
        np.testing.assert_allclose(fusion.predict_proba.call_args.args[0],
                                   np.concatenate([clinical, dermoscopic, metadata])[None, :])
        self.assertEqual(result["class_code"], "VASC")
        self.assertEqual(len(result["scores"]), 11)

    def test_missing_view_or_missing_model_never_fabricates_scores(self):
        with self.assertRaises(ValueError):
            self.predictor.analyze({"clinical": "c"}, mode="fusion")
        (self.directory / "best_clinical.pt").unlink()
        with patch.object(self.predictor, "_image_probs") as inference:
            with self.assertRaises(FileNotFoundError):
                self.predictor.analyze({"clinical": "c"})
            inference.assert_not_called()

    def test_nonfinite_or_malformed_scores_are_withheld(self):
        for scores in (np.full(11, np.nan), np.zeros(11), np.ones(10) / 10):
            with self.assertRaises(ValueError):
                self.predictor._probabilities(scores)

    def test_heatmap_failure_preserves_valid_prediction_and_targets_branch_choice(self):
        scores = np.full(11, .02);scores[7] = .8
        with patch.object(self.predictor, "_image_probs", return_value=scores), \
                patch.object(self.predictor, "_image_heatmap", side_effect=RuntimeError("explanation failed")) as heatmap:
            result = self.predictor.analyze({"clinical": "lesion"}, include_heatmaps=True)
        self.assertEqual(result["class_code"], "MEL")
        self.assertEqual(result["heatmaps"]["clinical"]["status"], "unavailable")
        self.assertIn("interpretation", result)
        heatmap.assert_called_once_with("lesion", "clinical", 7)


class ArtifactImportTests(unittest.TestCase):
    @staticmethod
    def bundle(*, missing=None, extra=None):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in REQUIRED_FILES:
                if name == missing:
                    continue
                value = json.dumps({"input_dim": 2, "numeric_means": {},
                                    "category_maps": {}, "feature_names": ["a", "b"]}) if name.endswith(".json") else "new checkpoint"
                archive.writestr("src/checkpoints/" + name, value)
            archive.writestr("src/checkpoints/clinical_resume.pt", "optimizer state, not for the UI")
            if extra:
                archive.writestr(extra, "unsafe")
        buffer.seek(0)
        return buffer

    def test_import_backs_up_existing_validator_and_skips_resume_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "checkpoints"; directory.mkdir()
            (directory / "input_validator.pt").write_text("existing validator")
            result = install_artifacts(self.bundle(), directory)
            self.assertEqual(set(result["installed"]), set(REQUIRED_FILES))
            self.assertFalse((directory / "clinical_resume.pt").exists())
            self.assertEqual((Path(result["backup"]) / "input_validator.pt").read_text(), "existing validator")

    def test_invalid_bundle_does_not_replace_existing_models(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            validator = directory / "input_validator.pt"; validator.write_text("keep")
            for archive in (self.bundle(missing="best_clinical.pt"), self.bundle(extra="../escape")):
                with self.assertRaises(ValueError):
                    install_artifacts(archive, directory)
                self.assertEqual(validator.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
