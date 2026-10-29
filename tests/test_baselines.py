import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.baselines import compare_metrics, freeze_baseline, verify_baseline
from utils.evaluation_protocol import evaluation_fingerprint, metric_protocol
from utils.model_artifacts import REQUIRED_FILES


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        (self.root / "src/checkpoints").mkdir(parents=True)
        (self.root / "src/example.py").write_text("print('baseline source')\n")
        self.rows = [{"lesion_id": f"lesion_{split}", "isic_id": f"image_{split}",
                      "image_type": "clinical: close-up", "target": "0", "split": split}
                     for split in ("train", "val", "fusion", "holdout")]
        self.csv = self.root / "split.csv"
        self.write_rows(self.rows)
        self.metrics = {"accuracy": 0.6, "balanced_accuracy": 0.4, "macro_f1": 0.3,
                        "fit_split": "fusion", "evaluation_split": "holdout", "evaluation_lesions": 1,
                        "fit_lesions": 1, "tta": False, "upload_gate_enabled": False,
                        "csv_sha256": hashlib.sha256(self.csv.read_bytes()).hexdigest()}
        self.metrics.update(metric_protocol(self.csv, "holdout"))

    def write_rows(self, rows):
        with self.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def artifact(self):
        artifact = Path(self.temp.name) / "artifacts.zip"
        with zipfile.ZipFile(artifact, "w") as archive:
            for name in REQUIRED_FILES:
                path = self.root / "src/checkpoints" / name
                path.write_bytes(b"baseline checkpoint")
                archive.write(path, "src/checkpoints/" + name)
            archive.writestr("src/checkpoints/best_fusion_lr.metrics.json", json.dumps(self.metrics))
            archive.write(self.csv, "data/processed/final_metadata_with_split.csv")
            for model in ("clinical", "dermoscopic", "metadata"):
                archive.writestr(f"src/checkpoints/{model}_resume.json", json.dumps({"complete": True, "epoch": 1,
                    "best_score": 0.3, "config": {"csv_sha256": self.metrics["csv_sha256"]}}))
        return artifact

    @staticmethod
    def unlock(directory):
        directory.chmod(0o755)
        for path in directory.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)

    def test_freeze_preserves_models_source_and_detects_tampering(self):
        artifact = self.artifact()
        directory = freeze_baseline(self.root, artifact, "baseline-test")
        self.addCleanup(self.unlock, directory)
        self.assertTrue(verify_baseline(directory)["intact"])
        self.assertFalse(directory.stat().st_mode & stat.S_IWUSR)
        with zipfile.ZipFile(directory / "payload/source_snapshot.zip") as archive:
            self.assertEqual(archive.read("src/example.py"), b"print('baseline source')\n")
            self.assertNotIn("src/checkpoints/best_clinical.pt", archive.namelist())
        (self.root / "src/checkpoints/best_clinical.pt").write_text("future model")
        self.assertTrue(verify_baseline(directory)["intact"])
        with self.assertRaises(FileExistsError):
            freeze_baseline(self.root, artifact, "baseline-test")
        metric_file = directory / "metrics.json"
        metric_file.chmod(0o644); metric_file.write_text("{}")
        with self.assertRaises(ValueError):
            verify_baseline(directory)

    def test_freeze_refuses_metrics_for_a_different_active_checkpoint(self):
        artifact = self.artifact()
        (self.root / "src/checkpoints/best_metadata.pt").write_text("another run")
        with self.assertRaises(ValueError):
            freeze_baseline(self.root, artifact, "wrong-run")
        self.assertFalse((self.root / "baselines/wrong-run").exists())

    def test_comparison_reports_deltas_and_refuses_protocol_changes(self):
        candidate = {**self.metrics, "macro_f1": 0.5}
        self.assertAlmostEqual(compare_metrics(self.metrics, candidate)["metrics"]["macro_f1"]["delta"], 0.2)
        for field, value in (("tta", True), ("decision_rule", "threshold-0.5"),
                             ("evaluation_fingerprint", "different-held-out-lesions"), ("upload_gate_enabled", True)):
            with self.assertRaises(ValueError):
                compare_metrics(self.metrics, {**candidate, field: value})
        with self.assertRaises(ValueError):
            compare_metrics(self.metrics, {**candidate, "accuracy": float("nan")})

    def test_training_rows_can_change_but_evaluation_inputs_cannot(self):
        before = evaluation_fingerprint(self.csv, "holdout")
        self.write_rows([*reversed(self.rows), {**self.rows[0], "lesion_id": "new_training_lesion"}])
        self.assertEqual(before, evaluation_fingerprint(self.csv, "holdout"))
        result = compare_metrics(self.metrics, {**self.metrics, "csv_sha256": "new-training-data"})
        self.assertTrue(result["training_csv_changed"])
        self.write_rows([{**row, "target": "1"} if row["split"] == "holdout" else row for row in self.rows])
        self.assertNotEqual(before, evaluation_fingerprint(self.csv, "holdout"))


if __name__ == "__main__":
    unittest.main()
