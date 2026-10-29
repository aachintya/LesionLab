"""Evaluate lesion-level fusion ensemble (stacked softmax + LogisticRegression)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(PROJECT_SRC / "components") not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC / "components"))


def _default_csv() -> Path:
    from components.paths import FINAL_METADATA_WITH_SPLIT_CSV

    return FINAL_METADATA_WITH_SPLIT_CSV


def evaluate_split(
    split_name: str,
    csv_path: Path,
    output_csv: Path | None = None,
    *,
    test_csv: Path | None = None,
    fusion_checkpoint: Path | None = None,
) -> dict:
    from components.paths import DIAGNOSIS_COLS, PROCESSED_TEST_METADATA_CSV
    from models.fusion_lr import load_fusion_lr, require_fusion_checkpoint
    from utils.ensemble import ensemble_predictions
    from utils.eval_inference import collect_lesion_modality_probs, require_checkpoints
    from utils.metrics import ZERO_DIVISION

    require_checkpoints(PROJECT_SRC)
    fusion_path = require_fusion_checkpoint(PROJECT_SRC, fusion_checkpoint)
    fusion_model = load_fusion_lr(fusion_path)

    inference = collect_lesion_modality_probs(
        split_name,
        csv_path,
        test_csv=test_csv or PROCESSED_TEST_METADATA_CSV,
        project_src=PROJECT_SRC,
    )
    lesion_ids = inference.lesion_ids

    ensemble_df = ensemble_predictions(
        lesion_ids,
        inference.clinical_probs,
        inference.dermoscopic_probs,
        inference.metadata_probs,
        fusion_model,
    )
    ensemble_df["pred_diagnosis"] = ensemble_df["pred_ensemble"].apply(
        lambda idx: DIAGNOSIS_COLS[int(idx)]
    )

    has_gt = inference.has_gt
    eval_metadata = inference.eval_metadata

    if has_gt:
        meta_by_lesion = eval_metadata.set_index("lesion_id")
        ensemble_df["target"] = meta_by_lesion.loc[lesion_ids, "target"].astype(int).values
        ensemble_df["diagnosis"] = meta_by_lesion.loc[lesion_ids, "diagnosis"].values
        y_true = ensemble_df["target"].astype(int).values
        y_ensemble = ensemble_df["pred_ensemble"].values
    else:
        ensemble_df["target"] = np.nan
        ensemble_df["diagnosis"] = ""
        y_true = y_ensemble = None

    results: dict = {
        "split": split_name,
        "n_lesions": len(lesion_ids),
        "data_source": (
            "MILK10k_Test_Metadata + MILK10k_Test_Input"
            if split_name == "test"
            else "final_metadata_with_split (val)"
        ),
        "has_ground_truth": has_gt,
    }

    if has_gt and y_true is not None:
        results.update(
            {
                "ensemble_accuracy": float(accuracy_score(y_true, y_ensemble)),
                "ensemble_macro_recall": float(
                    recall_score(y_true, y_ensemble, average="macro", zero_division=ZERO_DIVISION)
                ),
                "ensemble_macro_f1": float(
                    f1_score(y_true, y_ensemble, average="macro", zero_division=ZERO_DIVISION)
                ),
                "clinical_macro_recall": float(
                    recall_score(
                        inference.clinical_labels,
                        inference.clinical_preds,
                        average="macro",
                        zero_division=ZERO_DIVISION,
                    )
                ),
                "dermoscopic_macro_recall": float(
                    recall_score(
                        inference.dermoscopic_labels,
                        inference.dermoscopic_preds,
                        average="macro",
                        zero_division=ZERO_DIVISION,
                    )
                ),
                "metadata_macro_recall": float(
                    recall_score(
                        inference.metadata_labels,
                        inference.metadata_preds,
                        average="macro",
                        zero_division=ZERO_DIVISION,
                    )
                ),
            }
        )

    title = split_name.upper()
    if split_name == "test":
        title = "OFFICIAL MILK10k TEST"
    print(f"\n{'=' * 50}")
    print(f"  {title} — FUSION ENSEMBLE (logistic_regression)")
    print(f"{'=' * 50}")
    for key, value in results.items():
        if key != "split":
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    if not has_gt:
        print("\n  (No ground truth on official test set — metrics skipped.)")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        ensemble_df.to_csv(output_csv, index=False)
        print(f"\nSaved predictions -> {output_csv}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fusion LogisticRegression ensemble at lesion level."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_default_csv(),
        help="Training split CSV (used for val; class count for test).",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Processed test metadata (default: data/processed/test_metadata.csv).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="val = held-out training val; test = official MILK10k test release.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_SRC / "ensemble_predictions.csv",
        help="Output CSV with per-lesion ensemble predictions.",
    )
    parser.add_argument(
        "--fusion-checkpoint",
        type=Path,
        default=None,
        help="Path to best_fusion_lr.joblib (default: checkpoints/best_fusion_lr.joblib).",
    )
    args = parser.parse_args()

    if args.split == "val" and not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    evaluate_split(
        args.split,
        args.csv,
        output_csv=args.output,
        test_csv=args.test_csv,
        fusion_checkpoint=args.fusion_checkpoint,
    )


if __name__ == "__main__":
    main()
