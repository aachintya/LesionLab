"""Preserve the completed JI-ADF reference without changing the historical v1."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from ji_adf.runtime import write_json
from utils.baselines import compare_metrics, sha256_file, verify_baseline


def freeze_run(run_directory, destination, name="ji-adf-v1"):
    run, destination = Path(run_directory).resolve(), Path(destination).resolve()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name):
        raise ValueError("Use a simple lowercase baseline name")
    if destination.exists() or destination.is_relative_to(run) or run.is_relative_to(destination):
        raise ValueError("A new baseline needs a separate, non-existing destination")
    # Never add files underneath a previously frozen snapshot, even if its mode changed.
    if any((parent / "SHA256SUMS").exists() for parent in destination.parents):
        raise ValueError("Cannot nest a new baseline inside an existing frozen baseline")
    inference = json.loads((run / "inference.json").read_text())
    lock = json.loads((run / "evaluation_lock.json").read_text())
    comparison = json.loads((run / "comparison.json").read_text())
    metrics = json.loads((run / "metrics/ji_adf.holdout.json").read_text())
    if inference["variant"] != "ji_adf" or lock["primary"] != "ji_adf" or comparison["primary"] != "ji_adf":
        raise ValueError("Only the predeclared completed JI-ADF model can become this baseline")
    compare_metrics(json.loads((run / "v1.metrics.json").read_text()), metrics)
    expected_members = [member for job, member in lock["jobs"].items() if job.startswith("ji_adf/")]
    if inference["members"] != expected_members or not expected_members:
        raise ValueError("Inference members differ from the locked JI-ADF jobs")
    if sha256_file(run / "split.csv") != lock["csv_sha256"]:
        raise ValueError("The saved split CSV changed")
    from utils.evaluation_protocol import metric_protocol
    if metric_protocol(run / "split.csv", "holdout")["evaluation_fingerprint"] != metrics["evaluation_fingerprint"]:
        raise ValueError("The reported holdout does not match the saved split CSV")
    import pandas as pd
    from improved.training import metrics as score_predictions
    predictions = pd.read_csv(run / "predictions/ji_adf.holdout.csv")
    if len(predictions) != metrics["evaluation_lesions"] or predictions.lesion_id.duplicated().any():
        raise ValueError("The saved holdout predictions have invalid identities/count")
    from improved.data import paired_frame
    expected_targets = paired_frame(run / "split.csv").query("split == 'holdout'").set_index("lesion_id").target
    if (set(predictions.lesion_id) != set(expected_targets.index)
            or not (predictions.target.to_numpy() == expected_targets.loc[predictions.lesion_id].to_numpy()).all()):
        raise ValueError("Saved prediction identities/labels differ from the fixed holdout")
    observed = score_predictions(predictions.target.to_numpy(), predictions.predicted.to_numpy())
    if any(abs(observed[key] - metrics[key]) > 1e-10 for key in observed):
        raise ValueError("Recorded metrics differ from the saved predictions")
    required = {"inference.json", "evaluation_lock.json", "comparison.json", "comparison.md", "source_snapshot.zip",
                "environment.json", "run_plan.json", "reference_manifest.json", "input_validator.pt", "split.csv", "v1.metrics.json",
                "predictions/ji_adf.holdout.csv"}
    for member in inference["members"]:
        checkpoint = Path(member["checkpoint"])
        parent = checkpoint.parent
        # Paths come from run metadata, but still validate confinement before copying.
        for item in (member["checkpoint"], member["encoder"]):
            if not (run / item).resolve().is_relative_to(run):
                raise ValueError("Unsafe baseline member")
        status = json.loads((run / parent / "resume.json").read_text())
        if not status["complete"] or sha256_file(run / checkpoint) != member["checkpoint_sha256"]:
            raise ValueError("Cannot freeze incomplete or changed weights")
        config = json.loads((run / parent / "config.json").read_text())
        if status["config"] != config or config["variant"] != "ji_adf" or config["source_sha256"] != inference["source_sha256"]:
            raise ValueError("Frozen configuration differs from the completed JI-ADF job")
        if sha256_file(run / member["encoder"]) != member["encoder_sha256"]:
            raise ValueError("Cannot freeze a changed metadata encoder")
        required.update({member["checkpoint"], member["encoder"], str(parent / "config.json"), str(parent / "resume.json")})
    for item in required:
        if not (run / item).is_file():
            raise FileNotFoundError(run / item)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ji-adf-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for item in sorted(required):
            target = staging / item
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / item, target)
        write_json(staging / "metrics.json", metrics)
        (staging / "README.md").write_text(f"# {name}\n\nCompleted JI-ADF adaptation research baseline. Historical v1 remains separate.\n\n"
            f"Accuracy {metrics['accuracy']:.4f}; balanced accuracy {metrics['balanced_accuracy']:.4f}; macro F1 {metrics['macro_f1']:.4f}.\n"
            "Scores refer to the fixed internal holdout. See comparison.json for differences and uncertainty.\n"
            "No improvement or clinical validity is implied by designating a research baseline.\n"
            "This inference snapshot excludes optimizer tensors; keep the separate training artifact ZIP to resume.\n")
        files = {path.relative_to(staging).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                 for path in sorted(staging.rglob("*")) if path.is_file()}
        write_json(staging / "manifest.json", {"format_version": 3, "name": name, "status": "trained",
                   "architecture": "ji-adf-adaptation-v1", "source_sha256": inference["source_sha256"],
                   "historical_reference": "baseline-v1-2026-10-27", "metrics": metrics, "files": files})
        checks = [(sha256_file(path), path.relative_to(staging).as_posix()) for path in sorted(staging.rglob("*")) if path.is_file()]
        (staging / "SHA256SUMS").write_text("".join(f"{digest}  {item}\n" for digest, item in checks))
        os.rename(staging, destination)
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in destination.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o555)
    destination.chmod(0o555)
    verify_baseline(destination)
    return destination
