import contextlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image
from utils.input_validation import InputRejected, InputValidator
from scripts.train_input_validator import load_manifest, select_threshold


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "image.png"
        Image.new("RGB", (128, 128)).save(self.path)
        self.validator = InputValidator.__new__(InputValidator)
        self.validator.model = None

    def test_missing_validator_never_accepts(self):
        self.assertEqual(self.validator.validate(self.path).status, "validator_unavailable")
        with self.assertRaises(InputRejected):
            self.validator.require(self.path)

    def test_corrupt_image(self):
        self.path.write_text("not an image")
        self.assertEqual(self.validator.validate(self.path).status, "invalid_image")

    def test_small_image(self):
        Image.new("RGB", (30, 128)).save(self.path)
        self.assertEqual(self.validator.validate(self.path).status, "poor_quality")

    def test_learned_gate_and_nonfinite_score(self):
        class Tensor:
            def unsqueeze(self, _): return self
            def to(self, _): return self
            def item(self): return score

        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext, sigmoid=lambda x: x)
        self.validator.device = "cpu"
        self.validator.threshold = 0.8
        self.validator.transform = lambda _: Tensor()
        self.validator.model = lambda _: Tensor()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            for score, expected in [(0.1, "unsupported_image"), (0.9, "accepted"), (float("nan"), "validator_unavailable")]:
                result = self.validator.validate(self.path)
                self.assertEqual(result.status, expected)
                self.assertNotIn("diagnosis", result.to_dict())

    def test_threshold_rejects_negative_boundary(self):
        threshold = select_threshold([0.2, 0.7, 0.9, 0.95], [0, 0, 1, 1], 0)
        self.assertGreater(threshold, 0.7)
        self.assertLess(threshold, 0.9)

    def test_patient_overlap_rejected(self):
        manifest = Path(self.folder.name) / "manifest.csv"
        paths = []
        for i in range(6):
            path = Path(self.folder.name) / f"{i}.png"
            Image.new("RGB", (128, 128)).save(path)
            paths.append(path.name)
        manifest.write_text("path,label,split,group_id\n" + "\n".join(
            f"{paths[i]},{i % 2},{split},patient" for i, split in enumerate(["train", "train", "val", "val", "test", "test"])))
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
