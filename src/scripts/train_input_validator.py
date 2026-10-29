"""Train using CSV columns path,label,split,group_id; label 1=supported, 0=unrelated.

Splits must be train/val/test. Paths are relative to the CSV. Group by patient
for lesion photographs and by source/duplicate family for unrelated images.
Validation selects the rejection threshold; test is used only for reporting.
"""
import argparse
import json
import random
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_manifest(path):
    import pandas as pd
    df = pd.read_csv(path)
    required = {"path", "label", "split", "group_id"}
    if not required.issubset(df.columns) or df[list(required)].isna().any().any():
        raise ValueError("Manifest requires nonempty path,label,split,group_id columns")
    if not set(df.label).issubset({0, 1}) or not set(df.split).issubset({"train", "val", "test"}):
        raise ValueError("Labels must be 0/1; splits must be train/val/test")
    df["path"] = df.path.map(lambda p: str((path.parent / str(p)).resolve()))
    if df.path.duplicated().any() or not df.path.map(lambda p: Path(p).is_file()).all():
        raise ValueError("Image paths must exist and be unique")
    if (df.groupby("group_id").split.nunique() > 1).any():
        raise ValueError("A group occurs in multiple splits")
    hashes = df.path.map(lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    if "sha256" in df and not (df.sha256 == hashes).all():
        raise ValueError("Image contents do not match the manifest SHA256 hashes")
    if (df.assign(_hash=hashes).groupby("_hash").split.nunique() > 1).any():
        raise ValueError("Duplicate image contents occur in multiple splits")
    for split in ("train", "val", "test"):
        if set(df.loc[df.split == split, "label"]) != {0, 1}:
            raise ValueError(f"{split} requires both supported and unrelated examples")
    return df


def select_threshold(scores, labels, max_false_accept):
    """Lowest threshold satisfying the validation false-accept budget."""
    import numpy as np
    scores, labels = np.asarray(scores), np.asarray(labels)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite validation scores")
    candidates = sorted(set([1.0] + [float(np.nextafter(float(s), 1.0)) for s in scores[labels == 0]]))
    for threshold in candidates:
        if threshold > 0 and ((scores[labels == 0] >= threshold).mean() <= max_false_accept):
            return threshold
    raise ValueError("No usable threshold satisfies the false-accept budget")


def main():
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from torch.utils.data import Dataset, DataLoader
    from models.input_validator import build_input_validator, validator_transform
    from utils.input_validation import DEFAULT_VALIDATOR
    from utils.training import pick_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-false-accept", type=float, default=0.05)
    parser.add_argument("--linear-probe", action="store_true", help="Cache frozen pretrained features and train only the binary head (fast CPU baseline).")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or not 0 <= args.max_false_accept < 1:
        parser.error("epochs must be positive and max-false-accept must be in [0,1)")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    df = load_manifest(args.manifest.resolve())
    transform = validator_transform()

    class Images(Dataset):
        def __init__(self, split):
            self.rows = df[df.split == split].reset_index(drop=True)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows.iloc[index]
            with Image.open(row.path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                if min(image.size) < 64:
                    raise ValueError(f"Image smaller than inference minimum: {row.path}")
                return transform(image), torch.tensor(float(row.label))

    loaders = {s: DataLoader(Images(s), batch_size=16, shuffle=s == "train") for s in ("train", "val", "test")}
    device = pick_device()
    torch.hub.set_dir(str(Path(__file__).resolve().parents[1] / "checkpoints" / ".torch"))
    print(f"Training on {device}; linear_probe={args.linear_probe}", flush=True)
    model = build_input_validator(pretrained=True).to(device)
    score_loaders = loaders
    training_model = model
    if args.linear_probe:
        from torch import nn
        from torch.utils.data import TensorDataset
        extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten()).eval()
        cached = {}
        with torch.inference_mode():
            for split in ("train", "val", "test"):
                features, targets = [], []
                # Keep the manifest's order for saved per-image evaluation records.
                ordered = DataLoader(Images(split), batch_size=32, shuffle=False)
                for index, (images, labels) in enumerate(ordered):
                    features.append(extractor(images.to(device)).cpu())
                    targets.append(labels)
                    if index % 20 == 0:
                        print(f"Caching {split}: {min((index + 1) * 32, len(ordered.dataset))}/{len(ordered.dataset)}", flush=True)
                cached[split] = TensorDataset(torch.cat(features).clone(), torch.cat(targets))
        loaders = {s: DataLoader(cached[s], batch_size=64, shuffle=s == "train") for s in cached}
        score_loaders = {s: DataLoader(cached[s], batch_size=64, shuffle=False) for s in cached}
        training_model = model.fc
    optimizer = torch.optim.AdamW(training_model.parameters(), lr=0.003 if args.linear_probe else 1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        training_model.train()
        total = 0.0
        for images, labels in loaders["train"]:
            optimizer.zero_grad()
            loss = loss_fn(training_model(images.to(device)).flatten(), labels.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        print(f"epoch {epoch + 1}: loss={total / len(loaders['train'].dataset):.4f}", flush=True)

    def score(split):
        training_model.eval()
        scores, labels = [], []
        with torch.inference_mode():
            for images, target in score_loaders[split]:
                scores.extend(training_model(images.to(device)).flatten().sigmoid().cpu().tolist())
                labels.extend(target.tolist())
        return np.asarray(scores), np.asarray(labels)

    scores, labels = score("val")
    threshold = select_threshold(scores, labels, args.max_false_accept)
    def rate(k, n):
        # Wilson 95% interval exposes uncertainty from finite test samples.
        if n == 0:
            return None
        p, z = k / n, 1.959963984540054
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
        return {"rate": p, "count": n, "successes": k, "wilson_95": [max(0, center - radius), min(1, center + radius)]}

    report = {"threshold": threshold, "splits": {}, "by_source": {},
              "linear_probe": args.linear_probe, "epochs": args.epochs, "seed": 42,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "validation_max_false_accept": args.max_false_accept,
              "torch_version": str(torch.__version__), "device": str(device)}
    prediction_frames = []
    for split in ("val", "test"):
        scores, labels = score(split)
        records = df[df.split == split].copy()
        records["supported_probability"] = scores
        records["accepted"] = scores >= threshold
        prediction_frames.append(records)
        report["splits"][split] = {
            "unrelated_false_accept_rate": float((scores[labels == 0] >= threshold).mean()),
            "supported_accept_rate": float((scores[labels == 1] >= threshold).mean()),
            "supported_count": int((labels == 1).sum()),
            "unrelated_count": int((labels == 0).sum()),
            "unrelated_rejection": rate(int((scores[labels == 0] < threshold).sum()), int((labels == 0).sum())),
            "supported_acceptance": rate(int((scores[labels == 1] >= threshold).sum()), int((labels == 1).sum())),
        }
        if "source" in records:
            report["by_source"][split] = {
                source: rate(int((group.accepted == (group.label == 1)).sum()), len(group))
                for source, group in records.groupby("source")
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "threshold": threshold,
                "manifest_sha256": report["manifest_sha256"],
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, args.output)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2))
    import pandas as pd
    pd.concat(prediction_frames).to_csv(args.output.with_suffix(".predictions.csv"), index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
