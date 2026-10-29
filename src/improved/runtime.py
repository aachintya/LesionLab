"""Experiment isolation, provenance and safe DataLoader defaults."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import zipfile
from importlib.metadata import version
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.baselines import sha256_file


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hash():
    root = Path(__file__).resolve().parent
    scripts = root.parent / "scripts"
    paths = [*root.glob("*.py"), scripts / "train_improved.py", scripts / "evaluate_improved.py"]
    paths.extend(root.parent / path for path in ("utils/training.py", "utils/checkpointing.py",
                                                "components/metadata_encoder.py", "models/efficientnet_v2_image.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_reference(reference):
    reference = Path(reference).resolve()
    manifest = json.loads((reference / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = (reference / name).resolve()
        if not path.is_relative_to(reference) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen reference failed integrity check: {name}")
    return manifest


def output_directory(path, reference):
    path, reference = Path(path).resolve(), Path(reference).resolve()
    if ("baselines" in path.parts or path.is_relative_to(reference) or reference.is_relative_to(path)
            or ("src" in path.parts and "checkpoints" in path.parts)):
        raise ValueError("v2 outputs must be separate from the baseline and active app checkpoints")
    path.mkdir(parents=True, exist_ok=True)
    return path


def worker_init(_index):
    # Spawn avoids forking a multithreaded PyTorch/CUDA process.
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size, device, *, shuffle=False, workers=0):
    if workers < 0:
        raise ValueError("workers must be >=0")
    options = dict(num_workers=workers, pin_memory=device.type == "cuda")
    if workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True,
                       prefetch_factor=2, timeout=120, worker_init_fn=worker_init)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **options)


def move_batch(batch, device):
    result = {}
    for key in ("clinical", "dermoscopic", "metadata", "target"):
        value = batch[key].to(device, non_blocking=True)
        if key in ("clinical", "dermoscopic"):
            value = value.contiguous(memory_format=torch.channels_last)
        result[key] = value
    return result


def export_run(directory, destination):
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    if destination.is_relative_to(directory):
        raise ValueError("Export ZIP must be outside the run folder")
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if path.is_file() and not path.is_symlink() and "baseline_runtime" not in relative.parts and path.suffix != ".tmp":
                archive.write(path, relative)
    temporary.replace(destination)
    return destination


def capture_provenance(directory, reference):
    directory, reference = Path(directory), Path(reference)
    root = Path(__file__).resolve().parents[2]
    target = directory / "source_snapshot.zip"
    if not target.exists():
        temporary = target.with_suffix(".zip.tmp")
        files = [*root.glob("src/**/*.py"), root / "notebooks/kaggle_improved_v2.ipynb",
                 root / "requirements-kaggle-v2.txt", root / "docs/improved_v2.md"]
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(files):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
        temporary.replace(target)
    import shutil
    for source, name in ((reference / "metrics.json", "baseline.metrics.json"),
                         (reference / "manifest.json", "reference_manifest.json"),
                         (reference / "split.csv", "split.csv")):
        if not (directory / name).exists():
            shutil.copy2(source, directory / name)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", {"python": platform.python_version(),
                   "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow")},
                   "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   "source_sha256": source_hash()})
