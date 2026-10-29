"""Evaluate a frozen v2 ensemble once on the identical internal baseline holdout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate(args):
    import numpy as np
    import pandas as pd
    import torch
    from components.metadata_encoder import MetadataEncoder
    from improved.data import PairedDataset, paired_frame
    from improved.evaluation import detailed_metrics, ensemble_probabilities, paired_bootstrap, select_prior
    from improved.model import PairedLesionModel
    from improved.runtime import make_loader, output_directory, source_hash, verify_reference, write_json
    from improved.training import predict
    from utils.baselines import compare_metrics, sha256_file
    from utils.checkpointing import dataset_hash
    from utils.evaluation_protocol import metric_protocol
    from utils.training import pick_device

    reference = Path(args.reference).resolve()
    verify_reference(reference)
    directory = output_directory(args.output, reference)
    csv = reference / "split.csv"
    pairs = paired_frame(csv)
    counts = np.bincount(pairs.loc[pairs.split == "train", "target"], minlength=11)
    device = pick_device(args.device)
    torch.set_num_threads(2)
    seed_dirs = sorted(directory.glob("seed-*"))
    if not seed_dirs:
        raise FileNotFoundError("No seed outputs found")
    plan_path = directory / "run_plan.json"
    if plan_path.exists():
        expected = {f"seed-{seed}" for seed in json.loads(plan_path.read_text())["seeds"]}
        if {run.name for run in seed_dirs} != expected:
            raise ValueError("Seed output folders differ from the fixed run plan")
    checkpoints = {}
    configs = {}
    for run in seed_dirs:
        summary = json.loads((run / "resume.json").read_text())
        if not summary["complete"]:
            raise ValueError(f"Training is incomplete: {run}")
        config = json.loads((run / "config.json").read_text())
        if run.name != f"seed-{config['seed']}":
            raise ValueError("Seed configuration and output folder disagree")
        if config["source_sha256"] != source_hash() or config["csv_sha256"] != dataset_hash(csv):
            raise ValueError("Evaluation source or data differ from the trained run")
        if config["metadata_encoder_sha256"] != sha256_file(run / "metadata_encoder.json"):
            raise ValueError("The train-fitted metadata encoder changed")
        configs[run.name] = config
        checkpoints[run.name] = sha256_file(run / "best.pt")
    signature = {"checkpoints": checkpoints, "csv_sha256": dataset_hash(csv), "source_sha256": source_hash()}
    selection_path = directory / "selection.json"

    def logits_for(split):
        all_logits, identities, targets = [], None, None
        for run in seed_dirs:
            saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            config = saved["config"]
            if config != configs[run.name]:
                raise ValueError("Checkpoint and run settings disagree")
            encoder = MetadataEncoder.load(run / "metadata_encoder.json")
            dataset = PairedDataset(pairs[pairs.split == split], encoder, args.images, size=config["size"],
                                    cache_dir=args.cache, csv_sha256=config["csv_sha256"])
            model = PairedLesionModel(config["metadata_dim"]).to(device).to(memory_format=torch.channels_last)
            model.load_state_dict(saved["model"], strict=True)
            loader = make_loader(dataset, args.batch_size, device, workers=0)
            print(f"Predicting {split}: {run.name}, {len(dataset)} lesions", flush=True)
            ids, labels, logits = predict(model, loader, device)
            if identities is not None and (ids != identities or not np.array_equal(labels, targets)):
                raise ValueError("Seed predictions are not aligned")
            identities, targets = ids, labels
            all_logits.append(logits)
            del model, saved, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return identities, targets, np.stack(all_logits)

    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["signature"] != signature:
            raise ValueError("Selection is locked to different checkpoints. Use a NEW experiment directory.")
    else:
        _ids, labels, logits = logits_for("fusion")
        selection = select_prior(logits, labels, counts, split="fusion")
        selection["signature"] = signature
        selection["train_class_counts"] = counts.tolist()
        write_json(selection_path, selection)  # Persist BEFORE looking at holdout predictions.
    ids, labels, logits = logits_for("holdout")
    probabilities = ensemble_probabilities(logits, counts, selection["alpha"])
    predictions = pd.DataFrame({"lesion_id": ids, "target": labels, "predicted": probabilities.argmax(1)})
    from components.paths import DIAGNOSIS_COLS
    for index, name in enumerate(DIAGNOSIS_COLS):
        predictions["prob_" + name] = probabilities[:, index]
    predictions.to_csv(directory / "holdout_predictions.csv", index=False)
    report = {"model": "v2 paired feature fusion, equal-weight seed ensemble", "fit_split": "train",
              "selection_split": "val (checkpoint), fusion (scalar prior)", "evaluation_split": "holdout",
              "evaluation_lesions": len(labels), "fit_lesions": int((pairs.split == "train").sum()),
              "tta": False, "upload_gate_enabled": False, "csv_sha256": dataset_hash(csv),
              **metric_protocol(csv, "holdout"), **detailed_metrics(labels, probabilities)}
    write_json(directory / "holdout.metrics.json", report)
    import shutil
    shutil.copy2(reference / "checkpoints/input_validator.pt", directory / "input_validator.pt")
    shutil.copy2(reference / "manifest.json", directory / "reference_manifest.json")
    baseline = json.loads((reference / "metrics.json").read_text())
    comparison = compare_metrics(baseline, report)
    if args.reproduce_baseline:
        subprocess.run([sys.executable, str(Path(__file__).with_name("reproduce_baseline.py")),
                        "--reference", str(reference), "--images", str(args.images), "--output", str(directory)], check=True)
        original = pd.read_csv(directory / "baseline_holdout_predictions.csv").set_index("lesion_id")
        if set(original.index) != set(ids) or not np.array_equal(original.loc[ids, "target"].values, labels):
            raise ValueError("The baseline prediction IDs/labels do not match v2")
        comparison["paired_uncertainty"] = paired_bootstrap(labels, original.loc[ids, "predicted"].values,
                                                           probabilities.argmax(1), repetitions=args.bootstrap)
    else:
        comparison["paired_uncertainty"] = None
        comparison["note"] = "Aggregate comparison only. Reproduce the baseline for paired bootstrap intervals."
    write_json(directory / "comparison.json", comparison)
    rows = ["# v2 comparison with the frozen v1 baseline", "", f"Same {len(labels)}-lesion internal holdout; argmax; no TTA; curated inputs.",
            "", "| Metric | Baseline | v2 | Change |", "| --- | ---: | ---: | ---: |"]
    for name, values in comparison["metrics"].items():
        rows.append(f"| {name} | {values['baseline']:.4f} | {values['candidate']:.4f} | {values['delta']:+.4f} |")
    rows += ["", "Choose future changes on validation/fusion data. This internal holdout is already known; "
             "an untouched external evaluation is needed to establish generalization."]
    (directory / "comparison.md").write_text("\n".join(rows) + "\n")
    print(json.dumps(comparison, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--reproduce-baseline", action="store_true")
    p.add_argument("--bootstrap", type=int, default=2000)
    evaluate(p.parse_args())
