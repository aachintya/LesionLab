"""Export ISIC MILK10k Benchmark submission CSV via fusion LogisticRegression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(PROJECT_SRC / "components") not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC / "components"))


def _default_csv() -> Path:
    from components.paths import FINAL_METADATA_WITH_SPLIT_CSV

    return FINAL_METADATA_WITH_SPLIT_CSV


def _apply_calibration(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per-class monotonic recalibration: map each class's optimal threshold to 0.5.

    The leaderboard scores Dice/sensitivity one-vs-rest at 0.5. Our 11-class
    softmax sits well below 0.5 for the true class, so we piecewise-linearly
    stretch each column so that its Dice-optimal threshold t_c lands at 0.5.
    Monotonic -> preserves per-class AUC; output stays in [0, 1].
    """
    out = np.empty_like(probs)
    for c in range(probs.shape[1]):
        t = float(thresholds[c])
        col = probs[:, c]
        lo = col <= t
        out[lo, c] = 0.5 * col[lo] / max(t, 1e-6)
        out[~lo, c] = 0.5 + 0.5 * (col[~lo] - t) / max(1.0 - t, 1e-6)
    return np.clip(out, 0.0, 1.0)


def export_submission(
    split_name: str,
    csv_path: Path,
    output_path: Path,
    *,
    test_csv: Path | None = None,
    strict_lesion_count: bool = True,
    fusion_checkpoint: Path | None = None,
    tta: bool = False,
    calibrate: bool = False,
) -> pd.DataFrame:
    from components.paths import DIAGNOSIS_COLS, PROCESSED_TEST_METADATA_CSV
    from models.fusion_lr import fusion_prob_map, load_fusion_lr, require_fusion_checkpoint
    from utils.eval_inference import collect_lesion_modality_probs, require_checkpoints
    from utils.metrics import ZERO_DIVISION
    from utils.submission import (
        EXPECTED_BENCHMARK_LESIONS,
        build_submission_dataframe,
        validate_submission_dataframe,
    )

    require_checkpoints(PROJECT_SRC)
    fusion_path = require_fusion_checkpoint(PROJECT_SRC, fusion_checkpoint)
    fusion_model = load_fusion_lr(fusion_path)

    result = collect_lesion_modality_probs(
        split_name,
        csv_path,
        test_csv=test_csv or PROCESSED_TEST_METADATA_CSV,
        project_src=PROJECT_SRC,
        tta=tta,
    )

    fused_map = fusion_prob_map(
        fusion_model,
        result.lesion_ids,
        result.clinical_probs,
        result.dermoscopic_probs,
        result.metadata_probs,
    )
    probs = np.stack([fused_map[lid] for lid in result.lesion_ids], axis=0)

    if calibrate:
        thr_path = PROJECT_SRC / "checkpoints" / "_class_thresholds.npz"
        if not thr_path.exists():
            raise FileNotFoundError(
                f"Calibration thresholds not found: {thr_path}. "
                "Run: python scripts/calibrate_submission.py"
            )
        thresholds = np.load(thr_path)["thresholds"]
        probs = _apply_calibration(probs, thresholds)
        print("Applied per-class threshold calibration (optimal thresholds -> 0.5).")

    submission_df = build_submission_dataframe(result.lesion_ids, probs)
    validate_submission_dataframe(
        submission_df,
        split=split_name,
        strict_lesion_count=strict_lesion_count,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 50}")
    print(f"  ISIC SUBMISSION — {split_name.upper()}")
    print(f"{'=' * 50}")
    print(f"  lesions: {len(submission_df)}")
    print(f"  fusion: logistic_regression ({fusion_path.name})")
    if split_name == "test":
        print(f"  expected benchmark lesions: {EXPECTED_BENCHMARK_LESIONS}")

    if result.has_gt:
        meta_by_lesion = result.eval_metadata.set_index("lesion_id")
        y_true = meta_by_lesion.loc[result.lesion_ids, "target"].astype(int).values
        y_pred = submission_df[list(DIAGNOSIS_COLS)].to_numpy().argmax(axis=1)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=ZERO_DIVISION)
        print(f"  reference macro F1: {macro_f1:.4f}")

    print(f"\nSaved submission CSV -> {output_path}")
    return submission_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ISIC submission CSV (fusion LR probabilities)."
    )
    parser.add_argument("--csv", type=Path, default=_default_csv())
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Processed test metadata (default: data/processed/test_metadata.csv).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="test = official MILK10k Benchmark (479 lesions); val = held-out training val.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: milk10k_submission_test.csv or _val.csv).",
    )
    parser.add_argument(
        "--no-strict-count",
        action="store_true",
        help="Do not require exactly 479 lesions on test split.",
    )
    parser.add_argument(
        "--fusion-checkpoint",
        type=Path,
        default=None,
        help="Path to best_fusion_lr.joblib (default: checkpoints/best_fusion_lr.joblib).",
    )
    parser.add_argument(
        "--tta",
        dest="tta",
        action="store_true",
        default=True,
        help="Test-time augmentation (avg over flips/rotations). Default ON.",
    )
    parser.add_argument(
        "--no-tta", dest="tta", action="store_false", help="Disable test-time augmentation."
    )
    parser.add_argument(
        "--calibrate",
        dest="calibrate",
        action="store_true",
        default=True,
        help="Per-class threshold calibration (needs calibrate_submission.py). Default ON.",
    )
    parser.add_argument(
        "--no-calibrate",
        dest="calibrate",
        action="store_false",
        help="Disable calibration (raw softmax probabilities).",
    )
    args = parser.parse_args()

    if args.split == "val" and not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    if args.output is None:
        suffix = "test" if args.split == "test" else "val"
        args.output = PROJECT_SRC / f"milk10k_submission_{suffix}.csv"

    export_submission(
        args.split,
        args.csv,
        args.output,
        test_csv=args.test_csv,
        strict_lesion_count=not args.no_strict_count,
        fusion_checkpoint=args.fusion_checkpoint,
        tta=args.tta,
        calibrate=args.calibrate,
    )


if __name__ == "__main__":
    main()
