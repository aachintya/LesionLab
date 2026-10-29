"""Shared lesion-level inference for ensemble evaluation and submission export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


@dataclass
class LesionModalityProbs:
    lesion_ids: list[str]
    clinical_probs: dict[str, np.ndarray]
    dermoscopic_probs: dict[str, np.ndarray]
    metadata_probs: dict[str, np.ndarray]
    has_gt: bool
    eval_metadata: pd.DataFrame
    clinical_labels: np.ndarray
    clinical_preds: np.ndarray
    dermoscopic_labels: np.ndarray
    dermoscopic_preds: np.ndarray
    metadata_labels: np.ndarray
    metadata_preds: np.ndarray


def _checkpoint_dir(project_src: Path) -> Path:
    return project_src / "checkpoints"


def _num_classes(train_csv: Path) -> int:
    from components.paths import NUM_CLASSES

    return NUM_CLASSES


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def collect_lesion_modality_probs(
    split_name: str,
    csv_path: Path,
    *,
    test_csv: Path | None = None,
    project_src: Path | None = None,
    tta: bool = False,
    validate_inputs: bool = True,
) -> LesionModalityProbs:
    """Run clinical, dermoscopic, and metadata models; return per-lesion softmax maps."""
    from components.augmentations import get_transforms
    from components.data_utils import load_eval_split
    from components.dataset import MILK10kDataset
    from components.paths import PROCESSED_TEST_METADATA_CSV
    from models.efficientnet_v2_image import build_image_model
    from models.metadata_mlp import METADATA_CHECKPOINT_NAME
    from utils.inference import predictions_dict, run_inference
    from utils.metadata_inference import run_metadata_inference
    from utils.training import pick_device

    if project_src is None:
        project_src = Path(__file__).resolve().parents[1]

    device = pick_device()
    num_classes = _num_classes(csv_path)
    ckpt_dir = _checkpoint_dir(project_src)

    eval_clinical, eval_dermoscopic, eval_metadata, image_dir, has_gt = load_eval_split(
        split_name,
        train_csv=csv_path,
        test_csv=test_csv or PROCESSED_TEST_METADATA_CSV,
    )

    _, clinical_val_transform = get_transforms(modality="clinical")
    _, dermoscopic_val_transform = get_transforms(modality="dermoscopic")
    image_dir_str = str(image_dir)

    clinical_model = build_image_model(num_classes, dropout=0.5, pretrained=False).to(device)
    clinical_model.load_state_dict(_load_checkpoint(ckpt_dir / "best_clinical.pt", device))
    clinical_loader = DataLoader(
        MILK10kDataset(
            eval_clinical,
            image_dir_str,
            transform=clinical_val_transform,
            roi_mode="foreground",
            modality="clinical",
            cache_dir=os.environ.get("MILK10K_CACHE_DIR"),
        ),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    c_labels, c_preds, c_probs, c_ids = run_inference(
        clinical_model, clinical_loader, eval_clinical, id_col="lesion_id", device=device, tta=tta, validate_inputs=validate_inputs
    )

    dermoscopic_model = build_image_model(num_classes, dropout=0.5, pretrained=False).to(device)
    dermoscopic_model.load_state_dict(_load_checkpoint(ckpt_dir / "best_dermoscopic.pt", device))
    dermoscopic_loader = DataLoader(
        MILK10kDataset(
            eval_dermoscopic,
            image_dir_str,
            transform=dermoscopic_val_transform,
            roi_mode="foreground",
            modality="dermoscopic",
            cache_dir=os.environ.get("MILK10K_CACHE_DIR"),
        ),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    d_labels, d_preds, d_probs, d_ids = run_inference(
        dermoscopic_model, dermoscopic_loader, eval_dermoscopic, id_col="lesion_id", device=device,
        tta=tta, validate_inputs=validate_inputs,
    )

    m_labels, m_preds, m_probs, m_ids = run_metadata_inference(
        eval_metadata,
        encoder_path=ckpt_dir / "metadata_encoder.json",
        checkpoint_path=ckpt_dir / METADATA_CHECKPOINT_NAME,
    )

    _clinical_pred_map, clinical_prob_map = predictions_dict(c_ids, c_preds, c_probs)
    _dermoscopic_pred_map, dermoscopic_prob_map = predictions_dict(d_ids, d_preds, d_probs)
    _metadata_pred_map, metadata_prob_map = predictions_dict(m_ids, m_preds, m_probs)

    lesion_ids = sorted(set(c_ids) & set(d_ids) & set(m_ids))
    if len(lesion_ids) != len(c_ids):
        print(f"Warning: aligned {len(lesion_ids)} lesions (clinical had {len(c_ids)}).")

    return LesionModalityProbs(
        lesion_ids=lesion_ids,
        clinical_probs=clinical_prob_map,
        dermoscopic_probs=dermoscopic_prob_map,
        metadata_probs=metadata_prob_map,
        has_gt=has_gt,
        eval_metadata=eval_metadata,
        clinical_labels=c_labels,
        clinical_preds=c_preds,
        dermoscopic_labels=d_labels,
        dermoscopic_preds=d_preds,
        metadata_labels=m_labels,
        metadata_preds=m_preds,
    )


def require_checkpoints(project_src: Path) -> None:
    from models.metadata_mlp import METADATA_CHECKPOINT_NAME

    ckpt_dir = _checkpoint_dir(project_src)
    for name in (
        "best_clinical.pt",
        "best_dermoscopic.pt",
        METADATA_CHECKPOINT_NAME,
        "metadata_encoder.json",
    ):
        path = ckpt_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}. Run train_all_models.py first."
            )
