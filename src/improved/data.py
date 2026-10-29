"""Strict lesion pairing, train-only metadata encoding and inexpensive RGB inputs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from components.metadata_encoder import MetadataEncoder
from components.paths import NUM_CLASSES
from utils.checkpointing import dataset_hash

RGB_VERSION = "full-rgb-letterbox-v1"
META_COLUMNS = ("age_approx", "skin_tone_class", "sex", "site")
VIEWS = {"clinical: close-up": "clinical", "dermoscopic": "dermoscopic"}


def paired_frame(csv_path):
    frame = pd.read_csv(csv_path)
    required = {"lesion_id", "isic_id", "image_type", "target", "split", *META_COLUMNS}
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    if frame[list(required - set(META_COLUMNS))].isna().any().any():
        raise ValueError("Image identifiers, labels and splits cannot be missing")
    if set(frame.image_type) != set(VIEWS) or frame.isic_id.duplicated().any():
        raise ValueError("Expected unique image IDs and exactly the clinical/dermoscopic modalities")
    for name in ("lesion_id", "isic_id"):
        if not frame[name].astype(str).str.fullmatch(r"[A-Za-z0-9_-]+").all():
            raise ValueError(f"Invalid {name}")
    if not frame.target.isin(range(NUM_CLASSES)).all():
        raise ValueError("Targets must use the canonical 11-class order")
    rows = []
    for lesion, group in frame.groupby("lesion_id", sort=True):
        if len(group) != 2 or set(group.image_type) != set(VIEWS):
            raise ValueError(f"Lesion {lesion} must have exactly one image of each modality")
        for col in ("target", "split", *META_COLUMNS):
            if group[col].nunique(dropna=False) != 1:
                raise ValueError(f"Conflicting {col} for lesion {lesion}")
        row = group.iloc[0][["lesion_id", "target", "split", *META_COLUMNS]].to_dict()
        for kind, view in VIEWS.items():
            row[view + "_id"] = str(group.loc[group.image_type == kind, "isic_id"].iloc[0])
        rows.append(row)
    # Check any actual patient IDs before discarding unused source columns.
    for col in ("patient_id", "case_id", "subject_id", "patient"):
        if col in frame:
            known = frame[col].notna() & frame[col].astype(str).str.strip().ne("")
            if (frame[known].groupby(col).split.nunique() > 1).any():
                raise ValueError(f"{col} crosses splits")
    return pd.DataFrame(rows)


def encode_metadata(encoder, frame):
    # Numeric missingness is explicit; diagnostic labels, source and MONET never enter.
    missing = np.column_stack([pd.to_numeric(frame[c], errors="coerce").isna().astype(np.float32)
                               for c in ("age_approx", "skin_tone_class")])
    return np.concatenate([encoder.transform(frame[list(META_COLUMNS)]), missing], axis=1).astype(np.float32)


def fit_metadata(pairs, destination):
    train = pairs[pairs.split == "train"]
    if train.empty:
        raise ValueError("Cannot fit metadata without a train split")
    encoder = MetadataEncoder().fit(train[list(META_COLUMNS)])
    encoder.save(destination)
    return encoder


def original_path(image_root, lesion_id, image_id):
    root = Path(image_root)
    path = root / str(lesion_id) / f"{image_id}.jpg"
    if not path.is_file():
        path = root / f"{image_id}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def rgb_square(path, side):
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        return ImageOps.pad(rgb, (side, side), method=Image.Resampling.BICUBIC,
                            color=(124, 116, 104), centering=(0.5, 0.5))


def build_rgb_cache(csv_path, image_root, destination, *, side=384, workers=4):
    """Lossless resized PNGs; threads avoid the previous Linux fork deadlock."""
    if workers < 1 or side < 64:
        raise ValueError("Cache workers must be positive and side >=64")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pairs = paired_frame(csv_path)
    tasks = [(row.lesion_id, getattr(row, view + "_id"))
             for row in pairs.itertuples() for view in ("clinical", "dermoscopic")]

    def cache_one(task):
        lesion, identity = task
        source = original_path(image_root, lesion, identity)
        stat = source.stat()
        key = hashlib.sha256(f"{RGB_VERSION}:{side}:{identity}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:20]
        filename = f"{identity}-{key}.png"
        target = destination / filename
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            rgb_square(source, side).save(temporary, format="PNG", compress_level=1)
            temporary.replace(target)
        return identity, filename

    started = time.monotonic()
    index = {}
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for count, (identity, filename) in enumerate(pool.map(cache_one, tasks), 1):
            index[identity] = filename
            if count % 500 == 0 or count == len(tasks):
                print(f"RGB cache {count}/{len(tasks)} ({time.monotonic() - started:.1f}s)", flush=True)
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    payload = {"version": RGB_VERSION, "side": side, "csv_sha256": dataset_hash(csv_path), "images": index}
    temporary = destination / "index.json.tmp"
    temporary.write_text(json.dumps(payload))
    temporary.replace(destination / "index.json")
    return payload


def image_transform(size, training):
    operations = ([T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                   T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                   T.RandomRotation(180, fill=(124, 116, 104)),
                   T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)]
                  if training else [T.Resize((size, size), antialias=True)])
    return T.Compose([*operations, T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class PairedDataset(Dataset):
    def __init__(self, pairs, encoder, image_root, *, size=320, training=False,
                 cache_dir=None, csv_sha256=None):
        self.frame = pairs.reset_index(drop=True)
        self.metadata = torch.from_numpy(encode_metadata(encoder, self.frame))
        self.image_root = Path(image_root)
        self.transform = image_transform(size, training)
        self.side = max(384, size)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache = None
        if self.cache_dir:
            manifest = json.loads((self.cache_dir / "index.json").read_text())
            if (manifest["version"] != RGB_VERSION or manifest["side"] != self.side
                    or manifest["csv_sha256"] != csv_sha256):
                raise ValueError("RGB cache settings/data differ; rebuild it")
            self.cache = manifest["images"]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        images = []
        for view in ("clinical", "dermoscopic"):
            identity = row[view + "_id"]
            if self.cache is not None:
                with Image.open(self.cache_dir / self.cache[identity]) as image:
                    rgb = image.convert("RGB")
            else:
                rgb = rgb_square(original_path(self.image_root, row.lesion_id, identity), self.side)
            images.append(self.transform(rgb))
        return {"clinical": images[0], "dermoscopic": images[1],
                "metadata": self.metadata[index], "target": int(row.target), "lesion_id": str(row.lesion_id)}
