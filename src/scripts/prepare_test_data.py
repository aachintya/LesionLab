"""Prepare official MILK10k test metadata for inference (no ground-truth labels)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from components.paths import (  # noqa: E402
    PROCESSED_TEST_METADATA_CSV,
    TEST_METADATA_CSV,
    ensure_processed_dir,
    require_test_raw_files,
)


def prepare_test_metadata(
    *,
    metadata_path: Path = TEST_METADATA_CSV,
    output_path: Path = PROCESSED_TEST_METADATA_CSV,
) -> Path:
    require_test_raw_files()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Test metadata CSV not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if "age_approx" in df.columns:
        df["age_approx"] = df["age_approx"].fillna(df["age_approx"].median())

    for col in ("sex", "site"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    print("\n--- TEST SET REPORT ---")
    print(f"Total rows (images): {len(df)}")
    print(f"Unique lesions: {df['lesion_id'].nunique()}")
    print("Missing values filled (age_approx, sex, site).")
    print("Note: official test set has no diagnosis labels.")

    ensure_processed_dir()
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed test metadata -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MILK10k official test metadata CSV.")
    parser.add_argument("--metadata", type=Path, default=TEST_METADATA_CSV)
    parser.add_argument("--output", type=Path, default=PROCESSED_TEST_METADATA_CSV)
    args = parser.parse_args()
    prepare_test_metadata(metadata_path=args.metadata, output_path=args.output)


if __name__ == "__main__":
    main()
