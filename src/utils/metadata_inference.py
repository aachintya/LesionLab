"""Inference helpers for the metadata MLP model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from components.dataset import LesionMetadataDataset
from components.metadata_encoder import MetadataEncoder
from components.paths import NUM_CLASSES
from models.metadata_mlp import METADATA_CHECKPOINT_NAME, build_metadata_model
from utils.inference import run_inference
from utils.training import pick_device


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def run_metadata_inference(
    eval_df: pd.DataFrame,
    *,
    encoder_path: Path,
    checkpoint_path: Path | None = None,
    num_classes: int | None = None,
    batch_size: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if checkpoint_path is None:
        checkpoint_path = encoder_path.parent / METADATA_CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Metadata MLP checkpoint not found: {checkpoint_path}")

    if num_classes is None:
        num_classes = NUM_CLASSES

    device = pick_device()
    encoder = MetadataEncoder.load(encoder_path)
    model = build_metadata_model(
        encoder.input_dim_,
        num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(_load_checkpoint(checkpoint_path, device))
    model.eval()

    dataset = LesionMetadataDataset(eval_df, encoder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return run_inference(model, loader, eval_df, id_col="lesion_id", device=device)
