"""Freeze a local research baseline and compare matching evaluation reports."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import tempfile
import zipfile

from utils.evaluation_protocol import metric_protocol
from utils.model_artifacts import REQUIRED_FILES

METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _git(root, *arguments):
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True)
    return result.stdout if result.returncode == 0 else None


def _source_snapshot(root, output):
    listed = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if listed is None:
        candidates = [path for path in root.rglob("*") if path.is_file()]
    else:
        candidates = [root / os.fsdecode(name) for name in listed.split(b"\0") if name]
    excluded = {".git", ".venv", "data", "wandb", "outputs", "logs", "baselines", "dist"}
    inventory = {}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(candidates)):
            relative = path.relative_to(root)
            if (not path.is_file() or path.is_symlink() or relative.parts[0] in excluded
                    or relative.parts[:2] == ("src", "checkpoints")
                    or "__pycache__" in relative.parts or path.name.startswith(".env")
                    or relative.as_posix() == ".streamlit/secrets.toml"):
                continue
            data = path.read_bytes()
            archive.writestr(relative.as_posix(), data)
            inventory[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    return inventory


def freeze_baseline(root, artifacts, name):
    root, artifacts = Path(root).resolve(), Path(artifacts).resolve()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name):
        raise ValueError("Use a short lowercase baseline name containing letters, digits and hyphens.")
    directory = root / "baselines" / name
    if directory.exists():
        raise FileExistsError(f"Baseline already exists and will not be overwritten: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=directory.parent) as temp:
        staging = Path(temp)
        payload = staging / "payload"
        payload.mkdir()
        saved_zip = payload / "training_artifacts.zip"
        shutil.copy2(artifacts, saved_zip)
        with zipfile.ZipFile(saved_zip) as archive:
            if archive.testzip() is not None:
                raise ValueError("The Kaggle ZIP failed its CRC check.")
            required = {"src/checkpoints/" + item for item in REQUIRED_FILES}
            if not required <= set(archive.namelist()):
                raise ValueError("The Kaggle ZIP does not contain all inference checkpoints.")
            for member in required:
                with archive.open(member) as handle:
                    trained_hash = hashlib.file_digest(handle, "sha256").hexdigest()
                local = root / member
                if not local.is_file() or sha256_file(local) != trained_hash:
                    raise ValueError(f"The active checkpoint differs from the supplied Kaggle run: {member}")
            metrics = json.loads(archive.read("src/checkpoints/best_fusion_lr.metrics.json"))
            split_path = payload / "final_metadata_with_split.csv"
            split_path.write_bytes(archive.read("data/processed/final_metadata_with_split.csv"))
            if sha256_file(split_path) != metrics["csv_sha256"]:
                raise ValueError("The split CSV does not match the trained model's evaluation report.")
            training = {}
            for model in ("clinical", "dermoscopic", "metadata"):
                record = json.loads(archive.read(f"src/checkpoints/{model}_resume.json"))
                if not record["complete"] or record["config"]["csv_sha256"] != metrics["csv_sha256"]:
                    raise ValueError("Training is incomplete or a checkpoint uses different splits.")
                training[model] = {"epochs_completed": record["epoch"] + 1,
                                   "best_validation_score": record["best_score"], "config": record["config"]}
        metrics.update(metric_protocol(split_path, metrics["evaluation_split"]))
        write_json(staging / "metrics.json", metrics)
        with split_path.open(newline="") as handle:
            all_rows = list(csv.DictReader(handle))
        holdout = [row for row in all_rows if row["split"] == metrics["evaluation_split"]]
        if len({row["lesion_id"] for row in holdout}) != metrics["evaluation_lesions"]:
            raise ValueError("The evaluation lesion count does not match the saved report.")
        with (payload / "holdout_images.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(holdout[0]))
            writer.writeheader()
            writer.writerows(holdout)
        source_inventory = _source_snapshot(root, payload / "source_snapshot.zip")
        write_json(payload / "source_files.json", source_inventory)
        original_code = root / "dist/skin_cancer_kaggle.zip"
        if original_code.is_file():
            shutil.copy2(original_code, payload / "kaggle_code_bundle.zip")
        validator_reports = root / "src/checkpoints"
        for name_in in ("input_validator.metrics.json", "input_validator.api_test.json", "input_validator.bottle_stress.json"):
            source = validator_reports / name_in
            if source.is_file():
                shutil.copy2(source, payload / name_in)
        packages = {}
        for package in ("torch", "torchvision", "scikit-learn", "numpy", "pandas", "scipy", "Pillow", "opencv-python", "joblib", "streamlit"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        write_json(staging / "environment.json", {"scope": "local inference environment at baseline capture",
                   "python": platform.python_version(), "platform": platform.platform(), "packages": packages})
        raw = root / "data/raw"
        raw_files = ("MILK10k_Training_Metadata.csv", "MILK10k_Training_GroundTruth.csv", "MILK10k_Training_Supplement.csv",
                     ".downloads/MILK10k_Training_Input.zip")
        data_hashes = {item: sha256_file(raw / item) for item in raw_files if (raw / item).is_file()}
        readme = f"""# {name}

Frozen research baseline. The active app and checkpoints remain in the working project;
this directory is a separate, read-only snapshot for comparison and recovery.

| Metric | Internal holdout result |
| --- | ---: |
| Accuracy | {metrics['accuracy']:.2%} |
| Balanced accuracy | {metrics['balanced_accuracy']:.2%} |
| Macro F1 | {metrics['macro_f1']:.4f} |

Evaluation: {metrics['evaluation_lesions']} lesions, split `{metrics['evaluation_split']}`,
argmax decisions, TTA `{metrics['tta']}`, benchmark upload gate `{metrics['upload_gate_enabled']}`.
Normal UI inference keeps the input validator enabled. These are internal results.

- `payload/training_artifacts.zip`: the original Kaggle output, including best and
  resume checkpoints, metadata encoder, input validator, splits, histories and logs.
- `payload/source_snapshot.zip`: current source and GUI, including uncommitted files.
- `payload/kaggle_code_bundle.zip`, when present: the originally uploaded training bundle.
  The live Kaggle notebook was edited during setup; this is the original packaged notebook.
- `payload/final_metadata_with_split.csv`: exact split CSV from the completed run.
- `payload/holdout_images.csv`: fixed held-out image IDs, labels and metadata.
- `metrics.json`, `manifest.json`, `environment.json`: results, provenance and local dependencies.
- `SHA256SUMS`: integrity checks for every saved file except the checksum file itself.

Raw images remain in `data/raw` in the working project. Their official input ZIP and
CSV hashes are recorded in the manifest. The saved local dependency list is not a
complete lockfile for the original Kaggle GPU environment.

Verify: `python src/scripts/baseline.py verify baselines/{name}`

Compare a future evaluation: `python src/scripts/baseline.py compare --candidate
/path/to/new.metrics.json --baseline baselines/{name}` (put the command on one line).
The comparator requires matching held-out data, class order, decision rule, TTA and
gate policy. New training data can be used if the evaluation fingerprint stays fixed.
Tune on validation data; reserve the holdout for planned comparisons.

Recovery into a NEW folder: extract `source_snapshot.zip` and then
`training_artifacts.zip` there. Restore the shared raw dataset or point its environment
overrides to `data/raw`, then install `requirements-app.txt`. Resume optimizer files
are preserved; the UI importer intentionally installs inference files only.
"""
        (staging / "README.md").write_text(readme)
        git_head = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain")
        files = {path.relative_to(staging).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                 for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest = {"format_version": 1, "name": name, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "git_base_commit": git_head.decode().strip() if git_head else None,
                    "working_tree_was_dirty": bool(dirty), "source_file_count": len(source_inventory),
                    "metrics": metrics, "training": training, "raw_release_sha256": data_hashes, "files": files}
        write_json(staging / "manifest.json", manifest)
        checks = [(sha256_file(path), path.relative_to(staging).as_posix())
                  for path in sorted(staging.rglob("*")) if path.is_file()]
        (staging / "SHA256SUMS").write_text("".join(f"{digest}  {path}\n" for digest, path in checks))
        os.rename(staging, directory)
    # Separate copies and read-only modes guard against accidental future overwrites.
    for path in directory.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o555)
    directory.chmod(0o555)
    return directory


def verify_baseline(directory):
    directory = Path(directory)
    checked = 0
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Unsafe path in checksum file.")
        target = directory / name
        if not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unsafe target in checksum file.")
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"Baseline integrity check failed: {name}")
        checked += 1
    if not checked:
        raise ValueError("The baseline checksum file is empty.")
    return {"baseline": str(directory), "verified_files": checked, "intact": True}


def compare_metrics(baseline, candidate):
    fields = ("evaluation_split", "evaluation_lesions", "tta", "upload_gate_enabled", "decision_rule", "class_order")
    for key in (*fields, *METRICS):
        if key not in baseline or key not in candidate:
            raise ValueError(f"Missing evaluation metadata: {key}. Use the current train_fusion.py evaluation report.")
    mismatches = [key for key in fields if baseline[key] != candidate[key]]
    if baseline.get("evaluation_fingerprint") and candidate.get("evaluation_fingerprint"):
        if baseline["evaluation_fingerprint"] != candidate["evaluation_fingerprint"]:
            mismatches.append("held-out examples / metadata")
    elif not baseline.get("csv_sha256") or baseline["csv_sha256"] != candidate.get("csv_sha256"):
        mismatches.append("split CSV")
    if mismatches:
        raise ValueError("Results are not directly comparable; mismatched " + ", ".join(mismatches))
    result = {}
    for metric in METRICS:
        before, after = float(baseline[metric]), float(candidate[metric])
        if not (0 <= before <= 1 and 0 <= after <= 1):
            raise ValueError(f"Invalid {metric}; expected a finite value in [0, 1].")
        result[metric] = {"baseline": before, "candidate": after, "delta": after - before}
    return {"comparable": True, "metrics": result,
            "training_csv_changed": baseline.get("csv_sha256") != candidate.get("csv_sha256")}
