"""Paper-based objective and resumable validation-only checkpoint selection."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from components.metadata_encoder import MetadataEncoder
from components.paths import DIAGNOSIS_COLS
from improved.data import PairedDataset, RGB_VERSION, paired_frame, fit_metadata
from improved.training import metrics
from ji_adf.model import JIADFModel, PAPER_URL, PRECISION_POLICY, ji_adf_loss
from ji_adf.runtime import make_loader, move_batch, output_directory, source_hash, verify_reference, write_json
from utils.baselines import sha256_file
from utils.checkpointing import dataset_hash, restore_training, save_training
from utils.training import _autocast, atomic_torch_save, pick_device, seed_everything


def predict(model, loader, device):
    model.eval()
    identities, labels, probabilities, weights = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            moved = move_batch(batch, device)
            with _autocast(device, False):
                output = model(moved["clinical"], moved["dermoscopic"], moved["metadata"])
            if not all(torch.isfinite(value).all() for value in output.values()):
                raise RuntimeError(f"Nonfinite JI-ADF validation output for lesions {batch['lesion_id']}")
            probabilities.append(output["log_probs"].float().exp().cpu().numpy())
            weights.append(output["branch_weights"].float().cpu().numpy())
            identities.extend(batch["lesion_id"])
            labels.extend(batch["target"].tolist())
    return identities, np.asarray(labels), np.concatenate(probabilities), np.concatenate(weights)


def train(args, *, model_factory=JIADFModel):
    reference = Path(args.reference)
    manifest = verify_reference(reference)
    directory = output_directory(args.output, reference)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    csv = reference / "split.csv"
    pairs = paired_frame(csv)
    if set(pairs.split) != {"train", "val", "fusion", "holdout"}:
        raise ValueError("Use the frozen four-way split")
    encoder_path = directory / "metadata_encoder.json"
    encoder = MetadataEncoder.load(encoder_path) if encoder_path.exists() else fit_metadata(pairs, encoder_path)
    config = {key: getattr(args, key) for key in ("variant", "initialization", "seed", "epochs", "freeze_epochs", "patience",
                                                "batch_size", "accumulate", "size", "head_lr", "backbone_lr", "weight_decay",
                                                "workers", "cpu_threads")}
    config.update(format_version=3, architecture="ji-adf-adaptation-v1", paper=PAPER_URL,
                  source_sha256=source_hash(), csv_sha256=dataset_hash(csv), reference_sha256=manifest["sha256"],
                  metadata_dim=encoder.input_dim_ + 2, metadata_encoder_sha256=sha256_file(encoder_path),
                  preprocessing=RGB_VERSION, loss="NLL(final)+0.25CE(image)+0.5CE(joint)+0.25CE(metadata)",
                  optimizer="AdamW", scheduler="ReduceLROnPlateau(mode=max,factor=0.5,patience=3)",
                  batchnorm="frozen running statistics", amp=False,
                  precision=PRECISION_POLICY,
                  class_order=list(DIAGNOSIS_COLS))
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing JI-ADF configuration differs. Use a new experiment directory.")
    if (directory / "resume.pt").exists() and not args.resume:
        raise ValueError("An existing run needs --resume; overwrites are disabled")
    write_json(config_path, config)
    model = model_factory(config["metadata_dim"], variant=args.variant).to(device).to(memory_format=torch.channels_last)
    model.initialize(reference, args.initialization)
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith(("clinical.", "dermoscopic.")) else heads).append(parameter)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.backbone_lr}, {"params": heads, "lr": args.head_lr}],
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = restore_training(directory / "resume.pt", model, optimizer, scheduler, scaler, config, args.resume)
    if state["complete"]:
        print(f"{args.variant} seed {args.seed}: already complete", flush=True)
        return directory
    if state["epoch"] == -1:
        # Match data order/augmentation RNG across architectural controls, independent of module initialization counts.
        seed_everything(args.seed)
    common = dict(encoder=encoder, image_root=args.images, size=args.size, cache_dir=args.cache,
                  csv_sha256=config["csv_sha256"])
    train_data = PairedDataset(pairs[pairs.split == "train"], training=True, **common)
    validation = PairedDataset(pairs[pairs.split == "val"], **common)
    train_loader = make_loader(train_data, args.batch_size, device, shuffle=True, workers=args.workers)
    val_loader = make_loader(validation, args.batch_size, device, workers=args.workers)
    print(f"{args.variant}, seed {args.seed}: {len(train_data)} train / {len(validation)} validation lesions; "
          f"{device}; {args.size}px; microbatch {args.batch_size} x accumulation {args.accumulate}; init={args.initialization}", flush=True)
    for epoch in range(state["epoch"] + 1, args.epochs):
        started = time.monotonic()
        frozen = epoch < args.freeze_epochs
        model.freeze_encoders(frozen)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for index, batch in enumerate(train_loader):
            moved = move_batch(batch, device)
            group_start = (index // args.accumulate) * args.accumulate
            group_size = min(args.accumulate, len(train_loader) - group_start)
            with _autocast(device, False):
                outputs = model(moved["clinical"], moved["dermoscopic"], moved["metadata"])
                loss = ji_adf_loss(outputs, moved["target"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite JI-ADF loss at epoch {epoch+1}, batch {index+1}; "
                                   f"lesions {batch['lesion_id']}; stopped without replacing best checkpoint")
            scaler.scale(loss / group_size).backward()
            if (index + 1) % args.accumulate == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * len(batch["target"])
            if (index + 1) % 100 == 0 or index + 1 == len(train_loader):
                print(f"Epoch {epoch + 1}/{args.epochs} | batch {index + 1}/{len(train_loader)} | loss {loss.item():.4f}", flush=True)
        _ids, targets, probabilities, _weights = predict(model, val_loader, device)
        scores = metrics(targets, probabilities.argmax(1))
        elapsed = time.monotonic() - started
        state["history"].append({"epoch": epoch + 1, "train_loss": running_loss / len(train_data),
                                 "validation": scores, "seconds": elapsed,
                                 "learning_rates": [group["lr"] for group in optimizer.param_groups]})
        if scores["macro_f1"] > state["best_score"]:
            state["best_score"] = scores["macro_f1"]
            state["epochs_no_improve"] = 0
            atomic_torch_save({"format_version": 3, "model": model.state_dict(), "config": config,
                               "epoch": epoch + 1, "validation": scores}, directory / "best.pt")
        elif not frozen:
            state["epochs_no_improve"] += 1
        scheduler.step(scores["macro_f1"])
        state["epoch"] = epoch
        state["complete"] = epoch + 1 == args.epochs or state["epochs_no_improve"] >= args.patience
        save_training(directory / "resume.pt", state, model, optimizer, scheduler, scaler, config)
        print(f"EPOCH {epoch + 1} DONE | {elapsed:.1f}s | val macro F1 {scores['macro_f1']:.4f} | "
              f"val accuracy {scores['accuracy']:.4f} | best {state['best_score']:.4f}", flush=True)
        if state["complete"]:
            break
    return directory
