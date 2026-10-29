"""Lossless cache of deterministic preprocessing; augmentations stay online."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from components import preprocessing

_VERSION = hashlib.sha256(Path(preprocessing.__file__).read_bytes() + Path(__file__).read_bytes()).hexdigest()


def cache_path(image_path, cache_dir, modality, roi_mode):
    source = Path(image_path).resolve()
    stat = source.stat()
    identity = [str(source), stat.st_size, stat.st_mtime_ns, modality, roi_mode, _VERSION]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return Path(cache_dir) / modality / key[:2] / f"{key}.npy"


def load_preprocessed_image(image_path, *, modality, roi_mode="none", cache_dir=None):
    destination = cache_path(image_path, cache_dir, modality, roi_mode) if cache_dir else None
    if destination is not None and destination.exists():
        return np.load(destination, allow_pickle=False)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocessing.preprocess_for_modality(image, modality)
    if roi_mode == "foreground":
        image, _ = preprocessing.extract_lesion_roi_foreground(image)
    elif roi_mode != "none":
        raise ValueError(f"Unknown roi_mode: {roi_mode}")
    image = preprocessing.preprocess_final_steps(image)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.save(handle, image, allow_pickle=False)
                handle.flush()
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    return image
