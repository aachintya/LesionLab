"""Run inference for a single modality model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.training import forward_model, pick_device

# Test-time augmentation views for square lesion images: identity, h/v flips,
# and 90/180/270 rotations. Lesions are roughly orientation-invariant, so these
# are label-preserving; we average softmax over them to denoise predictions.
_TTA_OPS = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[3]),          # horizontal flip
    lambda x: torch.flip(x, dims=[2]),          # vertical flip
    lambda x: torch.rot90(x, 1, dims=[2, 3]),
    lambda x: torch.rot90(x, 2, dims=[2, 3]),
    lambda x: torch.rot90(x, 3, dims=[2, 3]),
]


def _unpack_image_batch(batch):
    """Return (image_tensor, labels) if this is an image batch, else (None, labels)."""
    if len(batch) == 2:
        inputs, labels = batch
        return inputs, labels
    if len(batch) == 3:
        first, labels, _extra = batch
        if torch.is_tensor(first) and first.dim() == 4:
            return first, labels
        return None, labels
    raise ValueError(f"Unexpected batch with {len(batch)} elements.")


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    dataframe: pd.DataFrame,
    *,
    id_col: str = "lesion_id",
    device: torch.device | None = None,
    tta: bool = False,
    validate_inputs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    device = device or pick_device()
    # Validate every original image before producing any disease probabilities.
    # Abort the entire batch on rejection so exports cannot silently lose rows.
    from utils.input_validation import InputValidator
    from components.dataset import LesionMetadataDataset
    if not isinstance(loader.dataset, LesionMetadataDataset):
        if not hasattr(loader.dataset, "_resolve_image_path"):
            raise TypeError("Image inference requires a dataset exposing original image paths")
        if validate_inputs:
            validator = InputValidator(device=device)
            for index in range(len(loader.dataset)):
                image_path, _ = loader.dataset._resolve_image_path(index)
                validator.require(image_path)
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    all_ids: list[str] = []

    row_idx = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = (_unpack_image_batch(batch) if tta else (None, None))
            if tta and inputs is not None and inputs.dim() == 4:
                inputs = inputs.to(device)
                labels = labels.to(device)
                prob_sum = None
                for op in _TTA_OPS:
                    p = torch.softmax(model(op(inputs)), dim=1)
                    prob_sum = p if prob_sum is None else prob_sum + p
                probs_t = prob_sum / len(_TTA_OPS)
                probs = probs_t.cpu().numpy()
                preds = probs_t.argmax(dim=1).cpu().numpy()
            else:
                outputs, labels = forward_model(model, batch, device)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()

            batch_size = labels.size(0)
            batch_ids = dataframe.iloc[row_idx : row_idx + batch_size][id_col].astype(str).tolist()
            row_idx += batch_size

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.append(probs)
            all_ids.extend(batch_ids)

    return (
        np.asarray(all_labels),
        np.asarray(all_preds),
        np.concatenate(all_probs, axis=0),
        all_ids,
    )


def predictions_dict(
    lesion_ids: list[str],
    preds: np.ndarray,
    probs: np.ndarray,
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    pred_map = {lid: int(p) for lid, p in zip(lesion_ids, preds)}
    prob_map = {lid: prob for lid, prob in zip(lesion_ids, probs)}
    return pred_map, prob_map
