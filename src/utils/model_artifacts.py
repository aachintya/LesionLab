"""Install only inference checkpoints from this project's own Kaggle output ZIP."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

REQUIRED_FILES = ("input_validator.pt", "best_clinical.pt", "best_dermoscopic.pt",
                  "best_metadata.pt", "metadata_encoder.json", "best_fusion_lr.joblib")
OPTIONAL_FILES = ("best_fusion_lr.metrics.json", "input_validator.metrics.json",
                  "clinical_resume.json", "dermoscopic_resume.json", "metadata_resume.json")


def install_artifacts(source, checkpoint_dir):
    """Stage and CRC-check selected files before replacing any current checkpoints.

    Import only your own training bundle: the existing fusion model uses joblib.
    Optimizer/resume tensors, source code, and datasets are not installed.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        selected = {}
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError("The ZIP contains an unsafe file path.")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("The ZIP contains a symbolic link.")
            if path.parent.as_posix() != "src/checkpoints" or path.name not in (*REQUIRED_FILES, *OPTIONAL_FILES):
                continue
            if path.name in selected:
                raise ValueError("The ZIP contains duplicate checkpoints.")
            limit = 2 * 1024**2 if path.suffix == ".json" else 192 * 1024**2
            if not 0 < member.file_size <= limit:
                raise ValueError(f"Invalid checkpoint size: {path.name}")
            selected[path.name] = member
        missing = set(REQUIRED_FILES) - selected.keys()
        if missing:
            raise ValueError("This training ZIP is missing: " + ", ".join(sorted(missing)))
        with tempfile.TemporaryDirectory(prefix=".model-import-", dir=checkpoint_dir.parent) as temp:
            staging = Path(temp)
            for name, member in selected.items():
                with archive.open(member) as incoming, (staging / name).open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024**2)
                if name.endswith(".json"):
                    json.loads((staging / name).read_text())
            encoder = json.loads((staging / "metadata_encoder.json").read_text())
            if not {"input_dim", "numeric_means", "category_maps", "feature_names"} <= encoder.keys():
                raise ValueError("The metadata encoder is not a valid project checkpoint.")
            existing = [name for name in selected if (checkpoint_dir / name).exists()]
            backup = None
            if existing:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                backup = checkpoint_dir / ".import-backups" / stamp
                backup.mkdir(parents=True)
                for name in existing:
                    shutil.copy2(checkpoint_dir / name, backup / name)
            replaced = []
            try:
                for name in selected:
                    os.replace(staging / name, checkpoint_dir / name)
                    replaced.append(name)
            except OSError:
                for name in replaced:
                    if name in existing:
                        shutil.copy2(backup / name, checkpoint_dir / name)
                    else:
                        (checkpoint_dir / name).unlink(missing_ok=True)
                raise
    return {"installed": sorted(selected), "backup": str(backup) if backup else None}
