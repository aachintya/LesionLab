"""Canonical MILK10k data paths (official release layout under data/raw/)."""

from __future__ import annotations

from pathlib import Path
import os

# src/components/paths.py -> repo root is two levels up from src/
_SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _SRC_ROOT.parent

RAW_DIR = Path(os.environ.get("MILK10K_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("MILK10K_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))

# Official MILK10k training release files
TRAINING_METADATA_CSV = RAW_DIR / "MILK10k_Training_Metadata.csv"
TRAINING_GROUND_TRUTH_CSV = RAW_DIR / "MILK10k_Training_GroundTruth.csv"
TRAINING_SUPPLEMENT_CSV = RAW_DIR / "MILK10k_Training_Supplement.csv"
TRAINING_INPUT_DIR = RAW_DIR / "MILK10k_Training_Input"

# Test release (no public ground truth in challenge set)
TEST_METADATA_CSV = RAW_DIR / "MILK10k_Test_Metadata.csv"
TEST_INPUT_DIR = RAW_DIR / "MILK10k_Test_Input"

# Processed pipeline outputs
FINAL_METADATA_CSV = PROCESSED_DIR / "final_metadata.csv"
FINAL_METADATA_WITH_SPLIT_CSV = PROCESSED_DIR / "final_metadata_with_split.csv"
PROCESSED_TEST_METADATA_CSV = PROCESSED_DIR / "test_metadata.csv"

NUM_CLASSES = 11

DIAGNOSIS_COLS = (
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
)


def training_image_dir() -> Path:
    """Root folder for training images: IL_<lesion_id>/ISIC_<id>.jpg."""
    return TRAINING_INPUT_DIR


def test_image_dir() -> Path:
    return TEST_INPUT_DIR


def ensure_processed_dir() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR


def require_test_raw_files() -> None:
    missing = [
        p
        for p in (TEST_METADATA_CSV, TEST_INPUT_DIR)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing MILK10k test files under data/raw/. Expected:\n"
            f"  - {TEST_METADATA_CSV.name}\n"
            f"  - {TEST_INPUT_DIR.name}/\n"
            f"Found missing: {[str(m) for m in missing]}"
        )


def require_training_raw_files() -> None:
    missing = [
        p
        for p in (
            TRAINING_METADATA_CSV,
            TRAINING_GROUND_TRUTH_CSV,
            TRAINING_INPUT_DIR,
        )
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing MILK10k training files under data/raw/. Expected:\n"
            f"  - {TRAINING_METADATA_CSV.name}\n"
            f"  - {TRAINING_GROUND_TRUTH_CSV.name}\n"
            f"  - {TRAINING_INPUT_DIR.name}/\n"
            f"Found missing: {[str(m) for m in missing]}"
        )
