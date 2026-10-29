"""JI-ADF provenance and shared safe runtime helpers; no changes to v2 files."""
from pathlib import Path
import hashlib
from importlib.metadata import version
import platform
import shutil
import zipfile

import torch

from improved.runtime import make_loader, move_batch, output_directory as _shared_output_directory, verify_reference, write_json, export_run


def output_directory(path, reference):
    target = Path(path).resolve()
    if any((parent / "SHA256SUMS").is_file() for parent in (target, *target.parents)):
        raise ValueError("Training/evaluation cannot write into a frozen baseline")
    return _shared_output_directory(target, reference)


def source_hash():
    src = Path(__file__).resolve().parents[1]
    files = [*src.glob("ji_adf/*.py"), src / "scripts/ji_adf.py"]
    files.extend(src / path for path in ("improved/data.py", "improved/runtime.py", "improved/evaluation.py",
                                         "improved/training.py", "components/metadata_encoder.py", "components/paths.py",
                                         "utils/training.py", "utils/checkpointing.py", "utils/evaluation_protocol.py",
                                         "utils/baselines.py", "utils/input_validation.py"))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(src).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def capture_provenance(directory, reference):
    directory, reference = Path(directory), Path(reference)
    root = Path(__file__).resolve().parents[2]
    destination = directory / "source_snapshot.zip"
    if not destination.exists():
        files = [*root.glob("src/**/*.py"), root / "notebooks/kaggle_ji_adf.ipynb",
                 root / "requirements-kaggle-ji-adf.txt", root / "docs/ji_adf.md"]
        temporary = destination.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(files):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
        temporary.replace(destination)
    for source, name in ((reference / "metrics.json", "v1.metrics.json"), (reference / "manifest.json", "reference_manifest.json"),
                         (reference / "split.csv", "split.csv"), (reference / "checkpoints/input_validator.pt", "input_validator.pt")):
        if not (directory / name).exists():
            shutil.copy2(source, directory / name)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", {"python": platform.python_version(), "source_sha256": source_hash(),
                   "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow")},
                   "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]})
