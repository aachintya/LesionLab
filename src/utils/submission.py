"""ISIC MILK10k Benchmark submission CSV helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from components.paths import DIAGNOSIS_COLS

LESION_ID_COLUMN = "lesion_id"
SUBMISSION_COLUMNS: tuple[str, ...] = (LESION_ID_COLUMN, *DIAGNOSIS_COLS)
EXPECTED_BENCHMARK_LESIONS = 479


def build_submission_dataframe(
    lesion_ids: Sequence[str],
    probs: np.ndarray,
) -> pd.DataFrame:
    """Build ISIC submission CSV from fused probabilities (n_lesions, n_classes)."""
    if probs.shape[0] != len(lesion_ids):
        raise ValueError(
            f"probs rows ({probs.shape[0]}) must match lesion count ({len(lesion_ids)})"
        )
    if probs.shape[1] != len(DIAGNOSIS_COLS):
        raise ValueError(
            f"Expected {len(DIAGNOSIS_COLS)} probability columns, got {probs.shape[1]}"
        )

    data: dict[str, list] = {LESION_ID_COLUMN: [str(lid) for lid in lesion_ids]}
    for col_idx, col_name in enumerate(DIAGNOSIS_COLS):
        data[col_name] = probs[:, col_idx].tolist()
    return pd.DataFrame(data, columns=list(SUBMISSION_COLUMNS))


def validate_submission_dataframe(
    df: pd.DataFrame,
    *,
    split: str,
    strict_lesion_count: bool = True,
) -> None:
    """Validate columns, probability range, and benchmark lesion count."""
    missing = [c for c in SUBMISSION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Submission CSV missing columns: {missing}")

    if df[LESION_ID_COLUMN].duplicated().any():
        dupes = df.loc[df[LESION_ID_COLUMN].duplicated(), LESION_ID_COLUMN].unique().tolist()
        raise ValueError(f"Duplicate lesion ids in submission: {dupes[:5]}")

    prob_cols = list(DIAGNOSIS_COLS)
    prob_arr = df[prob_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(prob_arr).all():
        raise ValueError("Submission contains non-finite probability values")
    if prob_arr.min() < 0.0 or prob_arr.max() > 1.0:
        raise ValueError(
            f"Probabilities must be in [0, 1]; got min={prob_arr.min():.6f}, max={prob_arr.max():.6f}"
        )

    n = len(df)
    if split == "test" and strict_lesion_count and n != EXPECTED_BENCHMARK_LESIONS:
        raise ValueError(
            f"Official benchmark expects {EXPECTED_BENCHMARK_LESIONS} lesions, got {n}. "
            "Check test metadata/images and checkpoint inference alignment."
        )
