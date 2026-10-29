"""Read-only inference for the locked frozen-feature/six-view ensemble.

The demo package contains copies of the selected downstream artifacts. It never
fits a classifier or writes to the original checkpoints or experiment archives.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from components.paths import DIAGNOSIS_COLS
from utils.feature_study import predict
from utils.frozen_ensemble import combine

ROOT = Path(__file__).resolve().parents[2]
ENSEMBLE_DIR = ROOT / "demo/ensemble-v1"
ENSEMBLE_ID = "geometric_feature_lr_tta_images"
TTA_VIEWS = ("identity", "horizontal_flip", "vertical_flip", "rotate90", "rotate180", "rotate270")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def missing_ensemble(directory=ENSEMBLE_DIR):
    return [name for name in ("manifest.json", "feature_classifier.joblib", "representations.joblib")
            if not (Path(directory) / name).is_file()]


def ensemble_signature(directory=ENSEMBLE_DIR):
    folder = Path(directory)
    return tuple((str(folder / name), (folder / name).stat().st_mtime_ns, (folder / name).stat().st_size)
                 for name in ("manifest.json", "feature_classifier.joblib", "representations.joblib")
                 if (folder / name).is_file())


def image_views(inputs):
    """Apply exactly the tensor transforms used by the saved experiment."""
    import torch
    return (inputs, torch.flip(inputs, [3]), torch.flip(inputs, [2]),
            torch.rot90(inputs, 1, [2, 3]), torch.rot90(inputs, 2, [2, 3]),
            torch.rot90(inputs, 3, [2, 3]))


class FrozenFeatureEnsemble:
    def __init__(self, checkpoint_dir, directory=ENSEMBLE_DIR):
        self.directory = Path(directory)
        missing = missing_ensemble(self.directory)
        if missing:
            raise FileNotFoundError("Improved ensemble package is incomplete: " + ", ".join(missing))
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if (self.manifest.get("id") != ENSEMBLE_ID
                or self.manifest.get("class_order") != list(DIAGNOSIS_COLS)
                or self.manifest.get("tta_views") != list(TTA_VIEWS)):
            raise ValueError("Improved ensemble identity or class order does not match this demo.")
        for name, digest in self.manifest["artifacts"].items():
            path = (self.directory / name).resolve()
            if (not path.is_relative_to(self.directory.resolve()) or not path.is_file()
                    or sha256(path) != digest):
                raise ValueError(f"Improved ensemble artifact failed integrity check: {name}")
        for name, digest in self.manifest["encoder_sha256"].items():
            if sha256(Path(checkpoint_dir) / name) != digest:
                raise ValueError(f"Improved ensemble requires its original trained checkpoint: {name}")
        for name, digest in self.manifest["preprocessing_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Improved ensemble preprocessing differs from its study: {name}")
        # Only the packaged, hash-verified project artifacts are deserialized.
        self.payload = joblib.load(self.directory / "feature_classifier.joblib")
        self.representations = joblib.load(self.directory / "representations.joblib")
        if (self.payload["class_order"] != list(DIAGNOSIS_COLS)
                or self.payload["spec"]["id"] != "pca64_plus_scores_lr_balanced_C0.1"
                or self.payload["encoder_files"] != self.manifest["encoder_sha256"]):
            raise ValueError("Improved feature classifier does not match the locked selection.")
        if list(self.payload["estimator"].classes_) != list(range(11)):
            raise ValueError("Improved feature classifier has an invalid class order.")

    def feature_vector(self, embeddings, metadata, scores):
        reduced = []
        for name in ("clinical", "dermoscopic"):
            z = np.asarray(embeddings[name], dtype=np.float32).reshape(1, -1)
            if z.shape != (1, 1280) or not np.isfinite(z).all():
                raise ValueError("Expected a finite 1280-dimensional image representation.")
            reduced.append(self.representations["reducers"][name].transform(z)[:, :64])
        metadata = np.asarray(metadata, dtype=np.float32).reshape(1, -1)
        if metadata.shape != (1, 12) or not np.isfinite(metadata).all():
            raise ValueError("Expected the original 12 encoded metadata features.")
        ordered = np.concatenate([scores[name] for name in ("clinical", "dermoscopic", "metadata")])
        vector = np.c_[*reduced, metadata, ordered.reshape(1, 33)]
        if vector.shape != (1, 173) or not np.isfinite(vector).all():
            raise ValueError("Invalid improved-classifier input.")
        return vector

    def predict(self, embeddings, metadata, scores, tta_scores):
        vector = self.feature_vector(embeddings, metadata, scores)
        feature = predict(self.payload["spec"], self.payload["estimator"], vector)
        images = (np.asarray(tta_scores["clinical"]) + np.asarray(tta_scores["dermoscopic"]))[None, :] / 2
        members = {"feature_lr": feature, "tta_images": images}
        spec = {"members": ["feature_lr", "tta_images"], "rule": "geometric"}
        return combine(spec, members)[0], {key: value[0] for key, value in members.items()}
