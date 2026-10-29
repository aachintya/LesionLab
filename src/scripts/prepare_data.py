"""Merge MILK10k training metadata + ground truth; impute; encode class labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from components.paths import (  # noqa: E402
    DIAGNOSIS_COLS,
    FINAL_METADATA_CSV,
    TRAINING_GROUND_TRUTH_CSV,
    TRAINING_METADATA_CSV,
    TRAINING_SUPPLEMENT_CSV,
    ensure_processed_dir,
    require_training_raw_files,
)


def prepare_final_metadata(
    *,
    metadata_path: Path = TRAINING_METADATA_CSV,
    ground_truth_path: Path = TRAINING_GROUND_TRUTH_CSV,
    supplement_path: Path | None = TRAINING_SUPPLEMENT_CSV,
    output_path: Path = FINAL_METADATA_CSV,
) -> Path:
    require_training_raw_files()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {ground_truth_path}")

    df_meta = pd.read_csv(metadata_path)
    df_truth = pd.read_csv(ground_truth_path)

    # Image-level metadata + lesion-level one-hot labels
    df_final = pd.merge(df_meta, df_truth, on="lesion_id", how="left")

    if supplement_path is not None and supplement_path.exists():
        df_sup = pd.read_csv(supplement_path)
        if "isic_id" in df_sup.columns:
            extra_cols = [c for c in df_sup.columns if c not in df_final.columns]
            if extra_cols:
                df_final = pd.merge(
                    df_final,
                    df_sup[["isic_id", *extra_cols]],
                    on="isic_id",
                    how="left",
                )

    total_images = len(df_final)
    unique_lesions = df_final["lesion_id"].nunique()

    print("\n--- VALIDATION REPORT ---")
    print(f"Total Rows (Images): {total_images} (Target: 10480)")
    print(f"Unique Lesion Count: {unique_lesions} (Target: 5240)")
    if total_images == 10480 and unique_lesions == 5240:
        print("Data counts are fully consistent with the report.")
    else:
        print("Warning: There is a mismatch in the counts!")

    # Leave missing values intact: the metadata encoder fits imputation on train only.
    existing_cols = [c for c in DIAGNOSIS_COLS if c in df_final.columns]
    if len(existing_cols) != len(DIAGNOSIS_COLS):
        raise ValueError("All 11 official diagnosis columns are required")
    labels = df_final[existing_cols].apply(pd.to_numeric, errors="coerce")
    if labels.isna().any().any() or not labels.isin([0, 1]).all().all() or not labels.sum(axis=1).eq(1).all():
        raise ValueError("Each image must join to exactly one valid one-hot ground-truth label")
    df_final["diagnosis"] = labels.idxmax(axis=1)
    df_final["target"] = df_final["diagnosis"].map({name: i for i, name in enumerate(DIAGNOSIS_COLS)})
    print("Class order:", dict(enumerate(DIAGNOSIS_COLS)))

    ensure_processed_dir()
    df_final.to_csv(output_path, index=False)
    print(f"\nSuccess! Cleaned data saved to:\n{output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MILK10k training metadata CSV.")
    parser.add_argument("--metadata", type=Path, default=TRAINING_METADATA_CSV)
    parser.add_argument("--ground-truth", type=Path, default=TRAINING_GROUND_TRUTH_CSV)
    parser.add_argument(
        "--supplement",
        type=Path,
        default=TRAINING_SUPPLEMENT_CSV,
        help="Optional supplement CSV (merged on isic_id). Pass empty to skip.",
    )
    parser.add_argument("--output", type=Path, default=FINAL_METADATA_CSV)
    args = parser.parse_args()

    supplement = args.supplement if str(args.supplement) else None
    prepare_final_metadata(
        metadata_path=args.metadata,
        ground_truth_path=args.ground_truth,
        supplement_path=supplement,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
