"""Stacking fusion: concat(clinical, dermoscopic, metadata) softmax -> LogisticRegression."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss

from components.paths import NUM_CLASSES
from utils.metrics import calculate_metrics, ZERO_DIVISION

FUSION_CHECKPOINT_NAME = "best_fusion_lr.joblib"
FUSION_FEATURE_DIM = NUM_CLASSES * 3


def resolve_fusion_checkpoint(
    project_src: Path,
    path: str | Path | None = None,
) -> Path:
    if path is not None:
        return Path(path)
    return project_src / "checkpoints" / FUSION_CHECKPOINT_NAME


def require_fusion_checkpoint(
    project_src: Path,
    path: str | Path | None = None,
) -> Path:
    ckpt = resolve_fusion_checkpoint(project_src, path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing fusion checkpoint: {ckpt}. "
            "Run: python scripts/train_fusion.py"
        )
    return ckpt


def build_fusion_feature_matrix(
    lesion_ids: Sequence[str],
    clinical_probs: dict[str, np.ndarray],
    dermoscopic_probs: dict[str, np.ndarray],
    metadata_probs: dict[str, np.ndarray],
    *,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Stack per-modality softmax vectors: shape (n_lesions, 3 * num_classes)."""
    rows: list[np.ndarray] = []
    for lesion_id in lesion_ids:
        parts = (
            clinical_probs[lesion_id],
            dermoscopic_probs[lesion_id],
            metadata_probs[lesion_id],
        )
        vec = np.concatenate(
            [np.asarray(p, dtype=np.float64).reshape(-1) for p in parts],
            axis=0,
        )
        if vec.shape[0] != num_classes * 3:
            raise ValueError(
                f"Expected {num_classes * 3} features for {lesion_id!r}, got {vec.shape[0]}"
            )
        rows.append(vec)
    return np.stack(rows, axis=0)


def default_fusion_lr() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        C=0.05,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )


def train_fusion_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    checkpoint_path: Path,
) -> LogisticRegression:
    """Fit balanced multinomial logistic regression on stacked modality probabilities."""
    model = default_fusion_lr()
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None and len(y_val) > 0:
        val_proba = model.predict_proba(X_val)
        val_pred = val_proba.argmax(axis=1)
        val_metrics = calculate_metrics(y_val, val_pred, y_prob=val_proba, average="macro")
        val_ll = log_loss(y_val, val_proba, labels=list(model.classes_))
        macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=ZERO_DIVISION)
        print(f"\n{'=' * 50}")
        print("  FUSION LR — VALIDATION")
        print(f"{'=' * 50}")
        print(f"  features: {X_val.shape[1]} (3 x {NUM_CLASSES} softmax)")
        print(f"  Val log loss: {val_ll:.4f}")
        print(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val macro recall: {val_metrics['recall']:.4f}")
        print(f"  Val macro F1: {macro_f1:.4f}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, checkpoint_path)
    print(f"Saved fusion LogisticRegression -> {checkpoint_path}")
    return model


def load_fusion_lr(path: str | Path) -> LogisticRegression:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, LogisticRegression):
        raise TypeError(f"Expected LogisticRegression in {path}, got {type(model)!r}")
    return model


def predict_fusion_proba(
    model: LogisticRegression,
    lesion_ids: Sequence[str],
    clinical_probs: dict[str, np.ndarray],
    dermoscopic_probs: dict[str, np.ndarray],
    metadata_probs: dict[str, np.ndarray],
) -> np.ndarray:
    """Return fused class probabilities, shape (n_lesions, n_classes)."""
    X = build_fusion_feature_matrix(
        lesion_ids,
        clinical_probs,
        dermoscopic_probs,
        metadata_probs,
    )
    return model.predict_proba(X)


def fusion_prob_map(
    model: LogisticRegression,
    lesion_ids: Sequence[str],
    clinical_probs: dict[str, np.ndarray],
    dermoscopic_probs: dict[str, np.ndarray],
    metadata_probs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    proba = predict_fusion_proba(
        model,
        lesion_ids,
        clinical_probs,
        dermoscopic_probs,
        metadata_probs,
    )
    return {lid: proba[i] for i, lid in enumerate(lesion_ids)}


def labels_for_lesions(eval_metadata: pd.DataFrame, lesion_ids: Sequence[str]) -> np.ndarray:
    meta = eval_metadata.set_index("lesion_id")
    return meta.loc[list(lesion_ids), "target"].astype(int).values
