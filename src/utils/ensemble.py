"""Lesion-level ensemble via fusion LogisticRegression on stacked modality softmax."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression


def ensemble_predictions(
    lesion_ids: list[str],
    clinical_probs: dict[str, np.ndarray],
    dermoscopic_probs: dict[str, np.ndarray],
    metadata_probs: dict[str, np.ndarray],
    fusion_model: "LogisticRegression",
    *,
    clinical_preds: dict[str, int] | None = None,
    dermoscopic_preds: dict[str, int] | None = None,
    metadata_preds: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Fuse three modality softmax vectors with trained fusion LR; return per-lesion preds."""
    from models.fusion_lr import predict_fusion_proba

    fused_proba = predict_fusion_proba(
        fusion_model,
        lesion_ids,
        clinical_probs,
        dermoscopic_probs,
        metadata_probs,
    )

    rows = []
    for i, lesion_id in enumerate(lesion_ids):
        c_prob = clinical_probs[lesion_id]
        d_prob = dermoscopic_probs[lesion_id]
        m_prob = metadata_probs[lesion_id]
        final_pred = int(np.argmax(fused_proba[i]))

        rows.append(
            {
                "lesion_id": lesion_id,
                "pred_clinical": (
                    clinical_preds[lesion_id]
                    if clinical_preds is not None
                    else int(np.argmax(c_prob))
                ),
                "pred_dermoscopic": (
                    dermoscopic_preds[lesion_id]
                    if dermoscopic_preds is not None
                    else int(np.argmax(d_prob))
                ),
                "pred_metadata": (
                    metadata_preds[lesion_id]
                    if metadata_preds is not None
                    else int(np.argmax(m_prob))
                ),
                "pred_ensemble": final_pred,
            }
        )
    return pd.DataFrame(rows)
