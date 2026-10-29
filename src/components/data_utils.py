"""Helpers for filtering MILK10k metadata by modality and lesion granularity."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

CLINICAL_IMAGE_TYPE = "clinical: close-up"
DERMOSCOPIC_IMAGE_TYPE = "dermoscopic"

MODALITY_IMAGE_TYPES = {
    "clinical": CLINICAL_IMAGE_TYPE,
    "dermoscopic": DERMOSCOPIC_IMAGE_TYPE,
}


def filter_by_image_type(df: "pd.DataFrame", image_type: str) -> "pd.DataFrame":
    if "image_type" not in df.columns:
        raise ValueError("DataFrame must contain an 'image_type' column.")
    out = df[df["image_type"] == image_type].copy()
    if out.empty:
        raise ValueError(f"No rows found for image_type={image_type!r}.")
    return out.reset_index(drop=True)


def filter_by_modality(df: "pd.DataFrame", modality: str) -> "pd.DataFrame":
    if modality not in MODALITY_IMAGE_TYPES:
        raise ValueError(
            f"Unknown modality {modality!r}. Expected one of: {list(MODALITY_IMAGE_TYPES)}"
        )
    return filter_by_image_type(df, MODALITY_IMAGE_TYPES[modality])


def lesion_level_metadata_df(df: "pd.DataFrame") -> "pd.DataFrame":
    """One row per lesion_id (metadata is identical across image types)."""
    if "lesion_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'lesion_id' column.")
    out = df.drop_duplicates(subset="lesion_id", keep="first").copy()
    return out.reset_index(drop=True)


def split_dataframe(df: "pd.DataFrame") -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """Return train, val, and legacy test rows from a training split CSV.

    The third frame is only non-empty if the CSV still has ``split=='test'`` rows
    (older pipelines). New ``make_splits.py`` outputs train/val only; use
    :func:`load_eval_split` with ``split='test'`` for official MILK10k test data.
    """
    if "split" not in df.columns:
        raise ValueError("DataFrame must contain a 'split' column.")
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def has_ground_truth(df: "pd.DataFrame") -> bool:
    return "target" in df.columns and df["target"].notna().all()


def load_eval_split(
    split: str,
    *,
    train_csv: "Path | None" = None,
    test_csv: "Path | None" = None,
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame", "Path", bool]:
    """Load clinical, dermoscopic, lesion-level metadata frames and image root for evaluation.

    - ``train`` / ``val``: rows from ``final_metadata_with_split.csv`` + training images.
    - ``test``: official MILK10k test release (``test_metadata.csv`` + ``MILK10k_Test_Input``).
    """
    from pathlib import Path

    from components.paths import (
        FINAL_METADATA_WITH_SPLIT_CSV,
        PROCESSED_TEST_METADATA_CSV,
        test_image_dir,
        training_image_dir,
    )

    if split in ("train", "val", "fusion", "holdout"):
        csv_path = Path(train_csv) if train_csv else FINAL_METADATA_WITH_SPLIT_CSV
        if not csv_path.exists():
            raise FileNotFoundError(f"Training split CSV not found: {csv_path}")
        import pandas as pd

        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].copy().reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No {split} rows found. Run make_splits.py first.")
        image_dir = training_image_dir()
    elif split == "test":
        csv_path = Path(test_csv) if test_csv else PROCESSED_TEST_METADATA_CSV
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Processed test metadata not found: {csv_path}. "
                "Run: python scripts/prepare_test_data.py"
            )
        import pandas as pd

        df = pd.read_csv(csv_path)
        image_dir = test_image_dir()
    else:
        raise ValueError(f"Unknown split {split!r}. Use 'train', 'val', or 'test'.")

    clinical_df = filter_by_modality(df, "clinical")
    dermoscopic_df = filter_by_modality(df, "dermoscopic")
    metadata_df = lesion_level_metadata_df(df)
    return clinical_df, dermoscopic_df, metadata_df, image_dir, has_ground_truth(metadata_df)
