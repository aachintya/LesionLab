"""Resumable two-phase joint training; selection sees validation, never holdout."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
import torch

from components.metadata_encoder import MetadataEncoder
from components.paths import NUM_CLASSES
from improved.data import PairedDataset, RGB_VERSION, fit_metadata, paired_frame
from improved.model import PairedLesionModel, balanced_loss
from improved.runtime import make_loader, move_batch, output_directory, source_hash, verify_reference, write_json
from utils.checkpointing import dataset_hash, restore_training, save_training
from utils.baselines import sha256_file
from utils.training import _autocast, atomic_torch_save, pick_device, seed_everything


def metrics(labels, predictions):
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(recall_score(labels, predictions, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)),
            "macro_f1": float(f1_score(labels, predictions, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0))}


def predict(model, loader, device):
    model.eval()
    all_logits, all_targets, identities = [], [], []
    with torch.inference_mode():
        for batch in loader:
            moved = move_batch(batch, device)
            with _autocast(device, device.type == "cuda"):
                outputs = model(moved["clinical"], moved["dermoscopic"], moved["metadata"])
            all_logits.append(outputs["logits"].float().cpu().numpy())
            all_targets.extend(batch["target"].tolist())
            identities.extend(batch["lesion_id"])
    return identities, np.asarray(all_targets), np.concatenate(all_logits)


def lr_multiplier(epoch, freeze_epochs, epochs):
    if epoch < freeze_epochs:
        return 1.0
    progress = (epoch - freeze_epochs) / max(1, epochs - freeze_epochs - 1)
    return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2


def train(args, *, model_factory=PairedLesionModel):
    reference = Path(args.reference)
    manifest = verify_reference(reference)
    directory = output_directory(args.output, reference)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    csv_path = reference / "split.csv"
    pairs = paired_frame(csv_path)
    encoder_path = directory / "metadata_encoder.json"
    # Each seed writes its own identical train-fitted encoder.
    if encoder_path.exists():
        encoder = MetadataEncoder.load(encoder_path)
    else:
        encoder = fit_metadata(pairs, encoder_path)
    metadata_dim = encoder.input_dim_ + 2
    config = {key: getattr(args, key) for key in ("seed", "epochs", "freeze_epochs", "patience", "batch_size",
                                                "size", "head_lr", "backbone_lr", "weight_decay", "accumulate")}
    config.update(format_version=2, source_sha256=source_hash(), csv_sha256=dataset_hash(csv_path),
                  reference_sha256=manifest["sha256"], metadata_dim=metadata_dim, preprocessing=RGB_VERSION,
                  metadata_encoder_sha256=sha256_file(encoder_path),
                  loss="balanced_softmax_aux0.2", batchnorm="frozen_statistics", amp=device.type == "cuda",
                  cpu_threads=args.cpu_threads, workers=args.workers)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing v2 configuration differs. Choose a NEW run directory.")
    if (directory / "resume.pt").exists() and not args.resume:
        raise ValueError("A run already exists. Use --resume or a NEW output directory.")
    write_json(config_path, config)
    model = model_factory(metadata_dim).to(device).to(memory_format=torch.channels_last)
    model.warm_start(reference / "checkpoints")
    backbone_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        (backbone_parameters if ".backbone." in name else head_parameters).append(parameter)
    optimizer = torch.optim.AdamW([{"params": backbone_parameters, "lr": args.backbone_lr},
                                   {"params": head_parameters, "lr": args.head_lr}], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_multiplier(epoch, args.freeze_epochs, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    state = restore_training(directory / "resume.pt", model, optimizer, scheduler, scaler, config, args.resume)
    if state["complete"]:
        print(f"Seed {args.seed} already complete", flush=True)
        return directory
    data_args = dict(encoder=encoder, image_root=args.images, size=args.size, cache_dir=args.cache,
                     csv_sha256=config["csv_sha256"])
    train_data = PairedDataset(pairs[pairs.split == "train"], training=True, **data_args)
    val_data = PairedDataset(pairs[pairs.split == "val"], **data_args)
    train_loader = make_loader(train_data, args.batch_size, device, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_data, args.batch_size, device, workers=args.workers)
    counts = torch.tensor(np.bincount(train_data.frame.target, minlength=NUM_CLASSES), dtype=torch.float32, device=device)
    if (counts <= 0).any():
        raise ValueError("Training requires every class")
    print(f"Seed {args.seed}: {len(train_data)} train / {len(val_data)} validation lesions, {device}, "
          f"{args.size}px, batch {args.batch_size} x accumulation {args.accumulate}", flush=True)
    for epoch in range(state["epoch"] + 1, args.epochs):
        started = time.monotonic()
        frozen = epoch < args.freeze_epochs
        model.freeze_encoders(frozen)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for index, batch in enumerate(train_loader):
            moved = move_batch(batch, device)
            # Correct normalization for an incomplete final accumulation group.
            group_start = (index // args.accumulate) * args.accumulate
            group_size = min(args.accumulate, len(train_loader) - group_start)
            with _autocast(device, device.type == "cuda"):
                outputs = model(moved["clinical"], moved["dermoscopic"], moved["metadata"])
                loss = balanced_loss(outputs, moved["target"], counts)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; stopped without overwriting best checkpoint")
            scaler.scale(loss / group_size).backward()
            if (index + 1) % args.accumulate == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * len(batch["target"])
            if (index + 1) % 50 == 0 or index + 1 == len(train_loader):
                print(f"Epoch {epoch + 1}/{args.epochs} {'fusion warmup' if frozen else 'joint fine tuning'} | "
                      f"batch {index + 1}/{len(train_loader)} | loss {loss.item():.4f}", flush=True)
        _ids, targets, logits = predict(model, val_loader, device)
        scores = metrics(targets, logits.argmax(1))
        elapsed = time.monotonic() - started
        record = {"epoch": epoch + 1, "phase": "warmup" if frozen else "finetune",
                  "train_loss": running_loss / len(train_data), "seconds": elapsed,
                  "validation": scores, "learning_rates": scheduler.get_last_lr()}
        state["history"].append(record)
        # Warmup scores do not prematurely stop the subsequent fine tuning.
        if scores["macro_f1"] > state["best_score"]:
            state["best_score"] = scores["macro_f1"]
            state["epochs_no_improve"] = 0
            atomic_torch_save({"model": model.state_dict(), "config": config,
                               "epoch": epoch + 1, "validation": scores}, directory / "best.pt")
        elif not frozen:
            state["epochs_no_improve"] += 1
        scheduler.step()
        state["epoch"] = epoch
        state["complete"] = epoch + 1 == args.epochs or state["epochs_no_improve"] >= args.patience
        save_training(directory / "resume.pt", state, model, optimizer, scheduler, scaler, config)
        print(f"EPOCH {epoch + 1} DONE | {elapsed:.1f}s | val macro F1 {scores['macro_f1']:.4f} | "
              f"val accuracy {scores['accuracy']:.4f} | best {state['best_score']:.4f}", flush=True)
        if state["complete"]:
            break
    return directory
