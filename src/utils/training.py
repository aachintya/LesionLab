"""Shared training utilities for image and metadata models."""

from __future__ import annotations

from typing import Any
from contextlib import nullcontext
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from utils.metrics import calculate_metrics


def pick_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            torch.cuda.set_device(device)
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    import cv2
    cv2.setNumThreads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader_options(num_workers: int, device: torch.device) -> dict:
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    options = dict(num_workers=num_workers, pin_memory=device.type == "cuda",
                   worker_init_fn=seed_worker)
    if num_workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return options


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        for index, saved in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(saved.cpu(), index)


def _autocast(device, enabled):
    return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def per_class_recall_f1(y_true, y_pred, *, num_classes: int, prefix: str) -> dict[str, float | int]:
    labels = list(range(num_classes))
    _precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0.0,
    )
    out: dict[str, float | int] = {}
    for c in labels:
        out[f"{prefix}_recall_c{c}"] = float(recall[c])
        out[f"{prefix}_f1_c{c}"] = float(f1[c])
        out[f"{prefix}_support_c{c}"] = int(support[c])
    return out


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    if hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = trainable
        if hasattr(model, "head"):
            for p in model.head.parameters():
                p.requires_grad = True
        return
    if hasattr(model, "features"):
        for p in model.features.parameters():
            p.requires_grad = trainable


def build_image_optimizer(
    model: nn.Module,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float = 1e-3,
) -> torch.optim.AdamW:
    """AdamW with lower LR on backbone (full finetune) and higher LR on head."""
    if not hasattr(model, "backbone") or not hasattr(model, "head"):
        return torch.optim.AdamW(model.parameters(), lr=head_lr, weight_decay=weight_decay)

    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def mixup_batch(images, labels, num_classes: int, alpha: float = 0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    idx = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[idx]
    labels_a, labels_b = labels, labels[idx]
    return mixed_images, labels_a, labels_b, lam


def forward_model(model: nn.Module, batch, device: torch.device, *, channels_last=False) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) not in (2, 3):
        raise ValueError(f"Unexpected batch with {len(batch)} elements.")
    inputs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
    if channels_last and inputs.ndim == 4:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    return model(inputs), labels


def _epoch_metrics(labels, predictions, num_classes, prefix):
    labels = torch.cat(labels).cpu().numpy()
    predictions = torch.cat(predictions).cpu().numpy()
    metrics = calculate_metrics(labels, predictions, y_prob=None, average="macro")
    metrics.update(per_class_recall_f1(labels, predictions, num_classes=num_classes, prefix=prefix))
    return metrics


def validate_one_epoch(model, loader, criterion, device, *, num_classes, amp=False, channels_last=False):
    model.eval()
    running_loss = torch.zeros((), device=device)
    all_labels, all_preds = [], []
    count = 0
    with torch.no_grad():
        for batch in loader:
            with _autocast(device, amp):
                outputs, labels = forward_model(model, batch, device, channels_last=channels_last)
                loss = criterion(outputs, labels)
            running_loss += loss.detach() * len(labels)
            count += len(labels)
            all_labels.append(labels.detach())
            all_preds.append(outputs.argmax(dim=1).detach())
    if not count:
        raise ValueError("Validation loader is empty")
    return float(running_loss.item() / count), _epoch_metrics(all_labels, all_preds, num_classes, "val")


def train_one_epoch(model, loader, criterion, optimizer, device, *, num_classes,
                    use_mixup=True, mixup_alpha=0.4, amp=False, scaler=None, channels_last=False):
    model.train()
    if hasattr(model, "backbone") and not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()  # Frozen batch-norm statistics must stay frozen too.
    running_loss = torch.zeros((), device=device)
    all_labels, all_preds = [], []
    count = 0
    for batch_idx, batch in enumerate(loader):
        inputs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        if channels_last and inputs.ndim == 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        mixed = use_mixup and inputs.ndim == 4
        if mixed:
            inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels, num_classes, alpha=mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            outputs = model(inputs)
            loss = (lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    if mixed else criterion(outputs, labels))
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        dominant = (labels_a if lam >= 0.5 else labels_b) if mixed else labels
        running_loss += loss.detach() * len(labels)
        count += len(labels)
        all_labels.append(dominant.detach())
        all_preds.append(outputs.argmax(dim=1).detach())
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}", flush=True)
    if not count:
        raise ValueError("Training loader is empty")
    return float(running_loss.item() / count), _epoch_metrics(all_labels, all_preds, num_classes, "train")


def print_epoch_metrics(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_metrics: dict[str, Any],
    val_loss: float,
    val_metrics: dict[str, Any],
) -> None:
    """Print train/val metrics in the same layout as the legacy training notebook."""
    print(f"\nEpoch {epoch}/{total_epochs}")
    print("Train Loss:", f"{train_loss:.4f}")
    print("Train Accuracy:", f"{train_metrics['accuracy']:.4f}")
    print("Train Recall:", f"{train_metrics['recall']:.4f}")
    print("Train F1:", f"{train_metrics['f1']:.4f}")
    print("Val Loss:", f"{val_loss:.4f}")
    print("Val Accuracy:", f"{val_metrics['accuracy']:.4f}")
    print("Val Recall:", f"{val_metrics['recall']:.4f}")
    print("Val F1:", f"{val_metrics['f1']:.4f}")
    print("Val Confusion Matrix:\n", val_metrics["confusion_matrix"])


def build_class_weights(train_df, device: torch.device, *, power: float = 0.5) -> torch.Tensor:
    """Per-class loss weights with tunable strength.

    power=1.0 is full inverse-frequency, 0.5 is sqrt-softened (legacy default),
    0.0 yields uniform weights (plain cross-entropy). When a balancing sampler is
    already active, keep the combined (sampler + class-weight) strength near 1.0.
    """
    class_counts = train_df["target"].value_counts().sort_index()
    raw_weights = (1.0 / class_counts.values.astype(float)) ** power
    normed = raw_weights / raw_weights.sum() * len(raw_weights)
    return torch.tensor(normed, dtype=torch.float32).to(device)
