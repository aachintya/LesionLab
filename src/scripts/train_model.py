"""Train a single modality model: clinical, dermoscopic, or metadata."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(PROJECT_SRC / "components") not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC / "components"))


def _default_csv() -> Path:
    from components.paths import FINAL_METADATA_WITH_SPLIT_CSV

    return FINAL_METADATA_WITH_SPLIT_CSV


def _checkpoint_dir() -> Path:
    path = PROJECT_SRC / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Clinical: longer schedule + lower backbone LR during full fine-tune.
CLINICAL_TRAIN_DEFAULTS: dict[str, int | float] = {
    "epochs": 30,
    "freeze_epochs": 6,
    "head_lr": 1e-4,
    "backbone_lr": 5e-5,
}


def train_modality(
    modality: str,
    *,
    csv_path: Path,
    epochs: int | None = None,
    batch_size: int = 16,
    lr: float = 1e-4,
    head_lr: float | None = None,
    backbone_lr: float | None = None,
    patience: int = 5,
    freeze_epochs: int | None = None,
    sampler_power: float = 0.5,
    class_weight_power: float = 0.5,
    select_metric: str = "f1",
    strong_aug_rare: bool = False,
    rare_threshold: int = 150,
    resume: bool = False,
    use_wandb: bool = False,
    device_name: str = "auto",
    num_workers: int = 0,
    cache_dir: Path | None = None,
    amp: bool = True,
    channels_last: bool = False,
    seed: int = 42,
    cpu_threads: int = 2,
) -> Path:
    if modality == "metadata":
        from scripts.train_metadata import train_metadata_mlp

        meta_epochs = 20 if epochs is None else epochs
        meta_lr = float(head_lr if head_lr is not None else lr)
        if head_lr is None and lr == 1e-4:
            meta_lr = 1e-3
        return train_metadata_mlp(
            csv_path,
            epochs=meta_epochs,
            batch_size=batch_size if batch_size >= 32 else 64,
            lr=meta_lr,
            patience=patience,
            sampler_power=sampler_power,
            class_weight_power=class_weight_power,
            select_metric=select_metric,
            ckpt_dir=_checkpoint_dir(),
            use_wandb=use_wandb,
            device_name=device_name, resume=resume, seed=seed,
        )

    from components.augmentations import get_mixup_alpha, get_transforms
    from components.data_utils import filter_by_modality, split_dataframe
    from components.dataset import MILK10kDataset
    from components.sampler import create_sampler
    from models.efficientnet_v2_image import build_image_model
    from utils.training import (
        build_class_weights,
        build_image_optimizer,
        pick_device,
        print_epoch_metrics,
        set_backbone_trainable,
        train_one_epoch,
        validate_one_epoch,
    )

    from utils.training import seed_everything, loader_options, atomic_torch_save
    from utils.checkpointing import dataset_hash, restore_training, save_training
    seed_everything(seed)
    df = pd.read_csv(csv_path)
    from components.paths import NUM_CLASSES
    num_classes = NUM_CLASSES
    ckpt_dir = _checkpoint_dir()

    if modality == "clinical":
        epochs = int(CLINICAL_TRAIN_DEFAULTS["epochs"] if epochs is None else epochs)
        freeze_epochs = int(
            CLINICAL_TRAIN_DEFAULTS["freeze_epochs"] if freeze_epochs is None else freeze_epochs
        )
        head_lr = float(CLINICAL_TRAIN_DEFAULTS["head_lr"] if head_lr is None else head_lr)
        backbone_lr = float(
            CLINICAL_TRAIN_DEFAULTS["backbone_lr"] if backbone_lr is None else backbone_lr
        )
    else:
        epochs = 20 if epochs is None else epochs
        freeze_epochs = 4 if freeze_epochs is None else freeze_epochs
        head_lr = lr if head_lr is None else head_lr
        backbone_lr = lr if backbone_lr is None else backbone_lr

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    torch.set_num_threads(cpu_threads)
    device = pick_device(device_name)
    amp = amp and device.type == "cuda"
    channels_last = channels_last and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if modality in ("clinical", "dermoscopic"):
        model_df = filter_by_modality(df, modality)
        train_df, val_df, _test_df = split_dataframe(model_df)
        train_transform, val_transform = get_transforms(modality=modality)
        from components.paths import training_image_dir

        image_dir = str(training_image_dir())
        mixup_alpha = get_mixup_alpha(modality)

        strong_transform = None
        strong_targets = None
        if strong_aug_rare:
            from components.augmentations import get_strong_transform

            counts = train_df["target"].value_counts()
            strong_targets = {int(c) for c in counts.index[counts < rare_threshold]}
            strong_transform = get_strong_transform(modality=modality)
            print(
                f"Class-aware strong aug for rare classes (train count < {rare_threshold}): "
                f"{sorted(strong_targets)}"
            )

        train_dataset = MILK10kDataset(
            train_df,
            image_dir,
            transform=train_transform,
            roi_mode="foreground",
            modality=modality,
            cache_dir=str(cache_dir) if cache_dir else None,
            strong_transform=strong_transform,
            strong_targets=strong_targets,
        )
        val_dataset = MILK10kDataset(
            val_df,
            image_dir,
            transform=val_transform,
            roi_mode="foreground",
            modality=modality,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        sampler = create_sampler(train_df, power=sampler_power)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            **loader_options(num_workers, device),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_options(num_workers, device))
        model = build_image_model(num_classes, dropout=0.5, pretrained=not (resume and (ckpt_dir / f"{modality}_resume.pt").exists())).to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        use_mixup = True
        checkpoint_path = ckpt_dir / f"best_{modality}.pt"
    else:
        raise ValueError(f"Unknown modality: {modality!r}")

    if set(train_df.target.astype(int)) != set(range(num_classes)):
        raise ValueError("Every diagnosis class must occur in the training split; check the dataset and group splits")
    class_weights = build_class_weights(train_df, device, power=class_weight_power)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    print(f"Balancing: sampler_power={sampler_power:g}, class_weight_power={class_weight_power:g}")
    if modality in ("clinical", "dermoscopic"):
        optimizer = build_image_optimizer(
            model, head_lr=head_lr, backbone_lr=backbone_lr, weight_decay=1e-3
        )
        print(
            f"Optimizer: head_lr={head_lr:g}, backbone_lr={backbone_lr:g}, "
            f"freeze_epochs={freeze_epochs}, epochs={epochs}"
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=head_lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="skin-cancer-detection",
            name=f"{modality}-model",
            config={
                "modality": modality,
                "batch_size": batch_size,
                "head_lr": head_lr,
                "backbone_lr": backbone_lr,
                "epochs": epochs,
                "freeze_epochs": freeze_epochs,
                "num_classes": num_classes,
            },
        )

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    resume_path = ckpt_dir / f"{modality}_resume.pt"
    config = dict(modality=modality, csv_sha256=dataset_hash(csv_path), epochs=epochs,
                  batch_size=batch_size, head_lr=head_lr, backbone_lr=backbone_lr,
                  freeze_epochs=freeze_epochs, select_metric=select_metric, seed=seed,
                  sampler_power=sampler_power, class_weight_power=class_weight_power,
                  patience=patience, strong_aug_rare=strong_aug_rare, rare_threshold=rare_threshold,
                  amp=amp, channels_last=channels_last)
    from components.image_cache import _VERSION as preprocessing_version
    config["preprocessing_version"] = preprocessing_version
    state = restore_training(resume_path, model, optimizer, scheduler, scaler, config, resume)
    best_score, epochs_no_improve = state["best_score"], state["epochs_no_improve"]
    start_epoch = state["epoch"] + 1
    if state["complete"]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Completed resume state requires best weights: {checkpoint_path}")
        return checkpoint_path

    print(f"Training {modality} model on {device}...")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        if modality in ("clinical", "dermoscopic"):
            set_backbone_trainable(model, trainable=(epoch >= freeze_epochs))
            if epoch >= freeze_epochs:
                phase = f"full finetune (head_lr={head_lr:g}, backbone_lr={backbone_lr:g})"
            else:
                phase = f"head-only (head_lr={head_lr:g}, backbone frozen)"
            print(f"[Phase] {phase}")
        else:
            print("[Phase] metadata model")

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=num_classes,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            amp=amp, scaler=scaler, channels_last=channels_last,
        )
        val_loss, val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes=num_classes,
            amp=amp, channels_last=channels_last,
        )
        scheduler.step()
        elapsed = time.perf_counter() - epoch_started
        print(f"Epoch time: {elapsed:.1f}s; AMP={amp}; channels_last={channels_last}", flush=True)

        print_epoch_metrics(
            epoch + 1,
            epochs,
            train_loss,
            train_metrics,
            val_loss,
            val_metrics,
        )

        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_metrics["accuracy"],
                    "train_recall": train_metrics["recall"],
                    "train_f1": train_metrics["f1"],
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "val_confusion_matrix": val_metrics["confusion_matrix"].tolist(),
                }
            )

        score = val_metrics[select_metric]
        if score > best_score:
            best_score = score
            atomic_torch_save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint (val {select_metric}={score:.4f}) -> {checkpoint_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        complete = epoch + 1 >= epochs or epochs_no_improve >= patience
        state.update(epoch=epoch, best_score=best_score, epochs_no_improve=epochs_no_improve, complete=complete)
        state["history"].append(dict(epoch=epoch + 1, seconds=elapsed, train_loss=train_loss,
                                   val_loss=val_loss, val_f1=float(val_metrics["f1"]),
                                   phase="finetune" if epoch >= freeze_epochs else "frozen"))
        save_training(resume_path, state, model, optimizer, scheduler, scaler, config)
        if complete:
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
            break

    if wandb_run is not None:
        import wandb

        wandb.finish()

    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one modality model for ensemble pipeline.")
    parser.add_argument(
        "--modality",
        required=True,
        choices=["clinical", "dermoscopic", "metadata"],
        help="Which model to train.",
    )
    parser.add_argument("--csv", type=Path, default=_default_csv())
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total epochs (clinical default: 30, others: 20).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Head LR for image models (clinical uses 1e-4 by default).",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="Override head learning rate (default: --lr, or clinical preset 1e-4).",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Backbone LR when unfrozen (clinical default: 5e-5).",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=None,
        help="Epochs with frozen backbone (clinical default: 6, dermoscopic: 4).",
    )
    parser.add_argument(
        "--sampler-power",
        type=float,
        default=0.5,
        help="Oversampling strength (1.0=full balance, 0.5=sqrt, 0.0=off).",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Loss class-weight strength (1.0=inverse-freq, 0.5=sqrt, 0.0=plain CE).",
    )
    parser.add_argument(
        "--select-metric",
        choices=["f1", "recall", "accuracy"],
        default="f1",
        help="Validation metric used to pick the best checkpoint.",
    )
    parser.add_argument(
        "--strong-aug-rare",
        action="store_true",
        help="Apply heavier (lesion-safe) augmentation to rare classes only.",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=150,
        help="Train-count below which a class gets strong augmentation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the per-epoch state file ({modality}_resume.pt) if present.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda:0, or cuda:1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    epochs = args.epochs
    freeze_epochs = args.freeze_epochs
    if args.modality == "clinical":
        if epochs is None:
            epochs = int(CLINICAL_TRAIN_DEFAULTS["epochs"])
        if freeze_epochs is None:
            freeze_epochs = int(CLINICAL_TRAIN_DEFAULTS["freeze_epochs"])
    else:
        if args.modality == "metadata":
            epochs = 20 if epochs is None else epochs
            freeze_epochs = 0
        else:
            epochs = 20 if epochs is None else epochs
            freeze_epochs = 4 if freeze_epochs is None else freeze_epochs

    path = train_modality(
        args.modality,
        csv_path=args.csv,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        patience=args.patience,
        freeze_epochs=freeze_epochs,
        sampler_power=args.sampler_power,
        class_weight_power=args.class_weight_power,
        select_metric=args.select_metric,
        strong_aug_rare=args.strong_aug_rare,
        rare_threshold=args.rare_threshold,
        resume=args.resume,
        use_wandb=args.wandb,
        device_name=args.device, num_workers=args.num_workers, cpu_threads=args.cpu_threads,
        cache_dir=args.cache_dir, amp=args.amp, channels_last=args.channels_last, seed=args.seed,
    )
    print(f"Done. Best model: {path}")


if __name__ == "__main__":
    main()
