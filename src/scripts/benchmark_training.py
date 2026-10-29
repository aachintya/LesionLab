"""Measure frozen/full fine-tuning throughput without saving model weights."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from components.augmentations import get_transforms
    from components.data_utils import filter_by_modality
    from components.dataset import MILK10kDataset
    from components.paths import NUM_CLASSES, training_image_dir
    from models.efficientnet_v2_image import build_image_model
    from utils.training import loader_options, set_backbone_trainable, pick_device, seed_everything

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_everything(42)
    torch.set_num_threads(2)
    device = pick_device(args.device)
    if device.type != "cuda" or args.steps < 1:
        raise ValueError("Benchmark requires CUDA and positive steps")
    torch.backends.cudnn.benchmark = True
    frame = pd.read_csv(args.csv)
    frame = filter_by_modality(frame[frame.split == "train"], "clinical")
    train_transform, _ = get_transforms(modality="clinical")
    dataset = MILK10kDataset(frame, str(training_image_dir()), transform=train_transform,
                            modality="clinical", roi_mode="foreground", cache_dir=str(args.cache_dir))
    loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True, **loader_options(args.num_workers, device))
    report = {"batch_size": args.batch_size, "device": torch.cuda.get_device_name(device), "phases": {}, "fits": True}
    try:
        model = build_image_model(NUM_CLASSES, dropout=0.5).to(device, memory_format=torch.channels_last)
        criterion = torch.nn.CrossEntropyLoss()
        for phase, trainable in [("frozen", False), ("finetune", True)]:
            set_backbone_trainable(model, trainable)
            model.train()
            if not trainable:
                model.backbone.eval()
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            warmup, measured, started = 2, 0, None
            for index, (images, labels) in enumerate(loader):
                if index >= warmup + args.steps:
                    break
                if index == warmup:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = criterion(model(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if index >= warmup:
                    measured += len(labels)
            if started is None:
                raise ValueError("Not enough training images for two warm-up batches plus measurement")
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            report["phases"][phase] = {"images_per_second": measured / elapsed,
                                       "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3}
    except torch.cuda.OutOfMemoryError:
        report["fits"] = False
        report["error"] = "GPU out of memory; choose a smaller batch size"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
