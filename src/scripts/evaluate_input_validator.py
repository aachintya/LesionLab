"""Evaluate the actual rejection API on a manifest without fitting anything."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import pandas as pd
    import torch
    from utils.input_validation import DEFAULT_VALIDATOR, InputValidator

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    validator = InputValidator(args.checkpoint)
    if validator.model is None:
        raise FileNotFoundError("Trained validator is required for evaluation")
    data = pd.read_csv(args.manifest)
    data = data[data.split == args.split].copy()
    if data.empty or not set(data.label).issubset({0, 1}):
        raise ValueError("Evaluation needs binary labels and a nonempty split")
    results = [validator.validate(args.manifest.parent / path) for path in data.path]
    data["status"] = [r.status for r in results]
    data["supported_probability"] = [r.supported_probability for r in results]
    data["accepted"] = [r.accepted for r in results]
    if not set(data.status).issubset({"accepted", "unsupported_image"}):
        raise ValueError("Evaluation encountered unreadable/undersized inputs or an unavailable validator")
    report = {"threshold": validator.threshold, "split": args.split, "images": len(data), "categories": {}}
    for category, group in data.groupby("category" if "category" in data else "source"):
        report["categories"][category] = {
            "count": len(group), "correct_decisions": int((group.accepted == (group.label == 1)).sum()),
            "false_accepts": int(((group.label == 0) & group.accepted).sum()),
            "false_rejects": int(((group.label == 1) & ~group.accepted).sum()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    data.to_csv(args.output.with_suffix(".predictions.csv"), index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
