"""Train metadata MLP (PyTorch) on lesion-level tabular features."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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


def train_metadata_mlp(
    csv_path: Path,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    sampler_power: float = 0.5,
    class_weight_power: float = 0.5,
    select_metric: str = "f1",
    ckpt_dir: Path | None = None,
    use_wandb: bool = False,
    device_name: str = "auto",
    resume: bool = False,
    seed: int = 42,
) -> Path:
    from components.data_utils import lesion_level_metadata_df, split_dataframe
    from components.dataset import LesionMetadataDataset
    from components.metadata_encoder import MetadataEncoder
    from components.sampler import create_sampler
    from models.metadata_mlp import (
        METADATA_CHECKPOINT_NAME,
        METADATA_ENCODER_NAME,
        build_metadata_model,
    )
    from utils.training import (
        build_class_weights,
        pick_device,
        print_epoch_metrics,
        train_one_epoch,
        validate_one_epoch,
    )

    from utils.training import seed_everything, atomic_torch_save
    from utils.checkpointing import dataset_hash, restore_training, save_training
    from components.paths import NUM_CLASSES
    seed_everything(seed)
    torch.set_num_threads(2)
    df = pd.read_csv(csv_path)
    num_classes = NUM_CLASSES
    ckpt_dir = ckpt_dir or _checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_df = lesion_level_metadata_df(df)
    train_df, val_df, _test_df = split_dataframe(model_df)

    if set(train_df.target.astype(int)) != set(range(num_classes)):
        raise ValueError("Every diagnosis class must occur in the training split")
    if len(train_df) < 2:
        raise ValueError("Metadata training requires at least two lesions")
    encoder = MetadataEncoder()
    encoder.fit(train_df)
    encoder.save(ckpt_dir / METADATA_ENCODER_NAME)

    train_dataset = LesionMetadataDataset(train_df, encoder)
    val_dataset = LesionMetadataDataset(val_df, encoder)
    sampler = create_sampler(train_df, power=sampler_power)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        drop_last=len(train_dataset) % batch_size == 1,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = pick_device(device_name)
    model = build_metadata_model(
        encoder.input_dim_,
        num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    checkpoint_path = ckpt_dir / METADATA_CHECKPOINT_NAME

    class_weights = build_class_weights(train_df, device, power=class_weight_power)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    print(
        f"Balancing: sampler_power={sampler_power:g}, class_weight_power={class_weight_power:g}"
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="skin-cancer-detection",
            name="metadata-mlp",
            config={
                "modality": "metadata",
                "batch_size": batch_size,
                "lr": lr,
                "epochs": epochs,
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
            },
        )

    resume_path = ckpt_dir / "metadata_resume.pt"
    config = dict(modality="metadata", csv_sha256=dataset_hash(csv_path), epochs=epochs,
                  batch_size=batch_size, lr=lr, patience=patience, hidden_dim=hidden_dim,
                  dropout=dropout, sampler_power=sampler_power, class_weight_power=class_weight_power,
                  select_metric=select_metric, seed=seed)
    state = restore_training(resume_path, model, optimizer, scheduler, None, config, resume)
    best_score, epochs_no_improve = state["best_score"], state["epochs_no_improve"]
    if state["complete"]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing best metadata weights: {checkpoint_path}")
        return checkpoint_path

    print(f"Training metadata MLP on {device}...")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"Input dim: {encoder.input_dim_}, checkpoint -> {checkpoint_path}")

    for epoch in range(state["epoch"] + 1, epochs):
        epoch_started = time.perf_counter()
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=num_classes,
            use_mixup=False,
        )
        val_loss, val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes=num_classes,
        )
        scheduler.step()
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
                    "val_loss": val_loss,
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
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
        state["history"].append(dict(epoch=epoch + 1, seconds=time.perf_counter() - epoch_started,
                                   train_loss=train_loss, val_loss=val_loss, val_f1=float(val_metrics["f1"])))
        save_training(resume_path, state, model, optimizer, scheduler, None, config)
        if complete:
            break

    if wandb_run is not None:
        import wandb

        wandb.finish()

    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train metadata MLP (standalone).")
    parser.add_argument("--csv", type=Path, default=_default_csv())
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    path = train_metadata_mlp(
        args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        sampler_power=args.sampler_power,
        class_weight_power=args.class_weight_power,
        select_metric=args.select_metric,
        use_wandb=args.wandb,
        device_name=args.device, resume=args.resume, seed=args.seed,
    )
    print(f"Done. Best model: {path}")


if __name__ == "__main__":
    main()
