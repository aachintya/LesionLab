"""Kaggle nested-archive upload regression and immutable-reference checks."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scripts.prepare_ji_adf_reference import prepare_reference, verify_reference


class ReferenceUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input/reference"
        self.source.mkdir(parents=True)
        self.destination = self.root / "working/reference_verified"
        contents = io.BytesIO()
        with zipfile.ZipFile(contents, "w") as archive:
            archive.writestr("src/old_model.py", "# frozen model source\n")
        self.snapshot = contents.getvalue()
        self.manifest = {"sha256": {"source_snapshot.zip": hashlib.sha256(self.snapshot).hexdigest(),
                                    "split.csv": hashlib.sha256(b"lesion,split\na,train\n").hexdigest()}}
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))
        (self.source / "split.csv").write_bytes(b"lesion,split\na,train\n")

    def test_missing_zip_restored_from_opaque_payload_without_changing_input(self):
        (self.source / "source_snapshot.zip.bin").write_bytes(self.snapshot)
        # Simulates Kaggle's transformation of the first upload.
        (self.source / "source_snapshot/src").mkdir(parents=True)
        (self.source / "source_snapshot/src/old_model.py").write_text("# unpacked\n")
        before = (self.source / "manifest.json").read_bytes()
        result = prepare_reference(self.source, self.destination)
        self.assertEqual(result, self.destination)
        self.assertEqual((result / "source_snapshot.zip").read_bytes(), self.snapshot)
        self.assertEqual(verify_reference(result), self.manifest)
        self.assertEqual((result / "manifest.json").read_bytes(), before)
        self.assertEqual((self.source / "manifest.json").read_bytes(), before)
        self.assertFalse((self.source / "source_snapshot.zip").exists())
        self.assertFalse((result / "source_snapshot").exists())
        self.assertEqual(prepare_reference(self.source, self.destination), result)

    def test_separate_repair_input_also_handles_zip_named_directory(self):
        (self.source / "source_snapshot.zip").mkdir()
        snapshot = self.root / "repair/source_snapshot.zip.bin"
        snapshot.parent.mkdir()
        snapshot.write_bytes(self.snapshot)
        prepare_reference(self.source, self.destination, snapshot=snapshot)
        self.assertEqual((self.destination / "source_snapshot.zip").read_bytes(), self.snapshot)
        self.assertTrue((self.source / "source_snapshot.zip").is_dir())

    def test_missing_or_bad_replacement_fails_before_publishing(self):
        with self.assertRaisesRegex(FileNotFoundError, "repair dataset"):
            prepare_reference(self.source, self.destination)
        (self.source / "source_snapshot.zip.bin").write_bytes(b"wrong archive")
        with self.assertRaisesRegex(ValueError, "expected SHA256"):
            prepare_reference(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_existing_corrupt_archive_cannot_be_hidden_by_valid_replacement(self):
        (self.source / "source_snapshot.zip").write_bytes(b"tampered archive")
        (self.source / "source_snapshot.zip.bin").write_bytes(self.snapshot)
        with self.assertRaisesRegex(ValueError, "expected SHA256"):
            prepare_reference(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_other_tampering_and_unsafe_paths_are_rejected(self):
        (self.source / "source_snapshot.zip").write_bytes(self.snapshot)
        (self.source / "split.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "split.csv"):
            prepare_reference(self.source, self.destination)
        self.manifest["sha256"] = {"../escaped": "irrelevant"}
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "Unsafe reference"):
            prepare_reference(self.source, self.destination)

    def test_frozen_or_input_destination_cannot_be_modified(self):
        (self.source / "source_snapshot.zip").write_bytes(self.snapshot)
        with self.assertRaisesRegex(ValueError, "separate"):
            prepare_reference(self.source, self.source / "new")
        frozen = self.root / "saved_baseline"
        frozen.mkdir()
        (frozen / "SHA256SUMS").write_text("frozen sentinel")
        with self.assertRaisesRegex(ValueError, "frozen baseline"):
            prepare_reference(self.source, frozen / "new")
        self.assertEqual((frozen / "SHA256SUMS").read_text(), "frozen sentinel")

    def test_corrupt_prepared_copy_is_rejected_and_kept_for_inspection(self):
        (self.source / "source_snapshot.zip").write_bytes(self.snapshot)
        prepare_reference(self.source, self.destination)
        (self.destination / "split.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "split.csv"):
            prepare_reference(self.source, self.destination)
        self.assertEqual((self.destination / "split.csv").read_text(), "changed")

    def test_builder_uses_upload_safe_archive_name_and_original_checksum(self):
        from scripts import build_ji_adf_bundle as builder
        from utils.model_artifacts import REQUIRED_FILES
        baseline = self.root / "baseline"
        (baseline / "payload").mkdir(parents=True)
        with zipfile.ZipFile(baseline / "payload/training_artifacts.zip", "w") as original:
            for name in REQUIRED_FILES:
                original.writestr("src/checkpoints/" + name, b"fixture")
        (baseline / "payload/source_snapshot.zip").write_bytes(self.snapshot)
        (baseline / "payload/final_metadata_with_split.csv").write_bytes(b"fixture split")
        (baseline / "metrics.json").write_text("{}")
        (baseline / "manifest.json").write_text(json.dumps({"raw_release_sha256": {}}))
        pretrained = self.root / "imagenet.pt"
        pretrained.write_bytes(b"fixture ImageNet weights")
        for name in ("README.md", "notebooks/kaggle_ji_adf.ipynb"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        bundle = self.root / "code.zip"
        with patch("utils.baselines.verify_baseline"), patch.object(builder, "IMAGENET_SHA256", hashlib.sha256(pretrained.read_bytes()).hexdigest()):
            builder.build(self.root, baseline, pretrained, bundle)
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            self.assertIn("reference/source_snapshot.zip.bin", names)
            self.assertNotIn("reference/source_snapshot.zip", names)
            self.assertFalse(any(name.endswith(".zip") for name in names))
            archive.extractall(self.root / "uploaded")
        prepared = prepare_reference(self.root / "uploaded/reference", self.root / "prepared")
        self.assertEqual((prepared / "source_snapshot.zip").read_bytes(), self.snapshot)


if __name__ == "__main__":
    unittest.main()
