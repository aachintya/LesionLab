"""Predeclared JI-ADF baseline, matched controls, and the unchanged v1 holdout."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from components.metadata_encoder import MetadataEncoder
from components.paths import DIAGNOSIS_COLS
from improved.data import PairedDataset, paired_frame
from improved.evaluation import detailed_metrics, paired_bootstrap
from ji_adf.model import JIADFModel, BRANCH_ORDER
from ji_adf.runtime import make_loader, output_directory, source_hash, verify_reference, write_json
from ji_adf.training import predict
from utils.baselines import compare_metrics, sha256_file
from utils.checkpointing import dataset_hash
from utils.evaluation_protocol import metric_protocol
from utils.training import pick_device


def locked_members(directory, reference):
    """Require completed planned jobs and lock their hashes before holdout inference."""
    directory, reference = Path(directory), Path(reference)
    plan = json.loads((directory / "run_plan.json").read_text())
    if plan["primary"] != "ji_adf" or "ji_adf" not in plan["variants"]:
        raise ValueError("The research baseline must be predeclared as ji_adf")
    expected = {f"{variant}/seed-{seed}" for variant in plan["variants"] for seed in plan["seeds"]}
    actual = {path.parent.relative_to(directory).as_posix() for path in directory.glob("*/seed-*/config.json")}
    if actual != expected:
        raise ValueError("Job folders differ from the predeclared run plan")
    groups = {variant: [] for variant in plan["variants"]}
    signatures = {}
    for name in sorted(expected):
        run = directory / name
        config = json.loads((run / "config.json").read_text())
        status = json.loads((run / "resume.json").read_text())
        if not status["complete"] or status["config"] != config:
            raise ValueError(f"Incomplete or mismatched training job: {name}")
        if config["source_sha256"] != source_hash() or config["csv_sha256"] != dataset_hash(reference / "split.csv"):
            raise ValueError("Source or split CSV differs from the trained configuration")
        if config["metadata_encoder_sha256"] != sha256_file(run / "metadata_encoder.json"):
            raise ValueError("Train-fitted metadata encoder changed")
        if name != f"{config['variant']}/seed-{config['seed']}":
            raise ValueError("Job settings and output folder disagree")
        member = {"checkpoint": f"{name}/best.pt", "encoder": f"{name}/metadata_encoder.json",
                  "checkpoint_sha256": sha256_file(run / "best.pt"),
                  "encoder_sha256": config["metadata_encoder_sha256"]}
        groups[config["variant"]].append(member)
        signatures[name] = member
    lock = {"primary": "ji_adf", "seed_weights": "equal", "calibration": "none", "tta": False,
            "selection": "validation macro F1 per job; architecture predeclared, no holdout selection",
            "source_sha256": source_hash(), "csv_sha256": dataset_hash(reference / "split.csv"), "jobs": signatures}
    path = directory / "evaluation_lock.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError("Evaluation is locked to other checkpoints/settings. Use a new experiment.")
    write_json(path, lock)
    return plan, groups, lock


def evaluate(args, *, model_factory=JIADFModel):
    reference = Path(args.reference).resolve()
    verify_reference(reference)
    directory = output_directory(args.output, reference)
    plan, groups, lock = locked_members(directory, reference)
    pairs = paired_frame(reference / "split.csv")
    device = pick_device(args.device)
    torch.set_num_threads(2)
    (directory / "metrics").mkdir(exist_ok=True)
    (directory / "predictions").mkdir(exist_ok=True)
    heldout = {}
    reports = {}
    for variant, members in groups.items():
        # These three reports expose generalization; none chooses the primary architecture.
        for split in ("val", "fusion", "holdout"):
            probabilities, branch_weights, ids, targets = [], [], None, None
            for member in members:
                saved = torch.load(directory / member["checkpoint"], map_location="cpu", weights_only=True)
                config = saved["config"]
                if config != json.loads((directory / member["checkpoint"]).with_name("config.json").read_text()):
                    raise ValueError("Checkpoint and configuration disagree")
                encoder = MetadataEncoder.load(directory / member["encoder"])
                dataset = PairedDataset(pairs[pairs.split == split], encoder, args.images, size=config["size"],
                                        cache_dir=args.cache, csv_sha256=config["csv_sha256"])
                loader = make_loader(dataset, args.batch_size, device, workers=0)
                model = model_factory(config["metadata_dim"], variant=variant).to(device).to(memory_format=torch.channels_last)
                model.load_state_dict(saved["model"], strict=True)
                print(f"Evaluating {variant}, {split}, {member['checkpoint']}: {len(dataset)} lesions", flush=True)
                identities, labels, prob, weights = predict(model, loader, device)
                if ids is not None and (identities != ids or not np.array_equal(labels, targets)):
                    raise ValueError("Seed IDs/labels are not aligned")
                ids, targets = identities, labels
                probabilities.append(prob)
                branch_weights.append(weights)
                del model, loader, saved
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            prob = np.mean(probabilities, axis=0)
            weights = np.mean(branch_weights, axis=0)
            if not np.isfinite(prob).all() or not np.allclose(prob.sum(1), 1, atol=1e-5):
                raise ValueError("Invalid model probabilities")
            report = {"model": f"JI-ADF adaptation v1 / {variant}", "fit_split": "train", "evaluation_split": split,
                      "evaluation_lesions": len(targets), "fit_lesions": int((pairs.split == "train").sum()),
                      "tta": False, "upload_gate_enabled": False, "decision_calibration": "none",
                      "seeds": plan["seeds"], "seed_aggregation": "equal probability mean", "primary": variant == "ji_adf",
                      "csv_sha256": dataset_hash(reference / "split.csv"), **metric_protocol(reference / "split.csv", split),
                      **detailed_metrics(targets, prob)}
            write_json(directory / "metrics" / f"{variant}.{split}.json", report)
            frame = pd.DataFrame({"lesion_id": ids, "target": targets, "predicted": prob.argmax(1)})
            for index, name in enumerate(DIAGNOSIS_COLS):
                frame["prob_" + name] = prob[:, index]
            for index, name in enumerate(BRANCH_ORDER):
                frame["weight_" + name] = weights[:, index]
            frame.to_csv(directory / "predictions" / f"{variant}.{split}.csv", index=False)
            if split == "holdout":
                heldout[variant] = frame
                reports[variant] = report
    original = json.loads((reference / "metrics.json").read_text())
    comparisons = {name: compare_metrics(original, report) for name, report in reports.items()}
    if args.reproduce_baseline:
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/reproduce_baseline.py"),
                   "--reference", str(reference), "--images", str(args.images), "--output", str(directory)]
        subprocess.run(command, check=True)
        baseline_pred = pd.read_csv(directory / "baseline_holdout_predictions.csv").set_index("lesion_id")
        for name, frame in heldout.items():
            ids = frame.lesion_id.tolist()
            if set(baseline_pred.index) != set(ids) or not np.array_equal(baseline_pred.loc[ids, "target"], frame.target):
                raise ValueError("v1 and JI-ADF evaluation examples differ")
            comparisons[name]["paired_uncertainty"] = paired_bootstrap(frame.target.to_numpy(), baseline_pred.loc[ids, "predicted"].to_numpy(),
                                                                       frame.predicted.to_numpy(), repetitions=args.bootstrap)
    else:
        for comparison in comparisons.values():
            comparison["paired_uncertainty"] = None
    control_comparisons = {}
    primary = heldout["ji_adf"].set_index("lesion_id")
    for name, frame in heldout.items():
        if name != "ji_adf":
            aligned = frame.set_index("lesion_id").loc[primary.index]
            if not np.array_equal(primary.target, aligned.target):
                raise ValueError("Control and primary labels differ")
            control_comparisons[name] = paired_bootstrap(primary.target.to_numpy(), aligned.predicted.to_numpy(), primary.predicted.to_numpy(),
                                                       repetitions=args.bootstrap)
    result = {"primary": "ji_adf", "new_research_baseline": "ji-adf-v1", "v1_comparisons": comparisons,
              "ji_adf_minus_controls": control_comparisons,
              "note": "The primary architecture was fixed before evaluation. Scores do not automatically replace the local app."}
    write_json(directory / "comparison.json", result)
    write_json(directory / "inference.json", {"format_version": 3, "architecture": "ji-adf-adaptation-v1", "variant": "ji_adf",
               "members": groups["ji_adf"], "seed_weights": "equal", "calibration": "none", "class_order": list(DIAGNOSIS_COLS),
               "source_sha256": lock["source_sha256"], "input_gate_required": True})
    rows = ["# JI-ADF research baseline v1", "", "Primary architecture: JI-ADF, fixed before evaluation.", "",
            "| Model | Accuracy | Balanced accuracy | Macro F1 |", "| --- | ---: | ---: | ---: |",
            f"| Saved v1 | {original['accuracy']:.4f} | {original['balanced_accuracy']:.4f} | {original['macro_f1']:.4f} |"]
    for name, report in reports.items():
        rows.append(f"| {name} | {report['accuracy']:.4f} | {report['balanced_accuracy']:.4f} | {report['macro_f1']:.4f} |")
    rows.extend(["", "Same internal held-out examples; argmax, no TTA. The full JSON includes paired uncertainty.",
                 "The rarest held-out class has one lesion. No external or clinical superiority is established."])
    (directory / "comparison.md").write_text("\n".join(rows) + "\n")
    print((directory / "comparison.md").read_text(), flush=True)
    return result
