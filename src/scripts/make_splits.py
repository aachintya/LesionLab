from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


# Prefer true patient / case identifiers; "attribution" is a coarse proxy for data source.
_DEFAULT_GROUP_CANDIDATES = (
    "patient_id",
    "case_id",
    "subject_id",
    "patient",
)


def _resolve_group_column(df: pd.DataFrame, group_col: Optional[str]) -> str:
    if group_col is not None:
        if group_col not in df.columns:
            raise ValueError(
                f"group_col={group_col!r} not found. Available: {list(df.columns)}"
            )
        return group_col
    for c in _DEFAULT_GROUP_CANDIDATES:
        if c in df.columns and df[c].notna().any():
            return c
    return "lesion_id"


def _split_group_key_series(df: pd.DataFrame, group_col: str) -> pd.Series:
    """One group id per row; missing patient id falls back to lesion_id so rows are not merged."""
    base = df[group_col].astype("string")
    out = base.copy()
    missing = base.isna() | (base.str.strip() == "")
    if "lesion_id" in df.columns:
        out = out.mask(missing, df["lesion_id"].astype("string"))
    else:
        out = out.mask(missing, pd.RangeIndex(len(df)).astype("string"))
    return out


def _group_table_for_stratify(df: pd.DataFrame, split_group: pd.Series) -> pd.DataFrame:
    work = df[["target"]].copy()
    work["_split_group"] = split_group.values
    # Representative label per group: first mode of target (handles multi-diagnosis patients).
    def _mode_int(s: pd.Series) -> int:
        m = s.mode()
        return int(m.iloc[0]) if len(m) else int(s.iloc[0])

    g = (
        work.groupby("_split_group", observed=False)["target"]
        .agg(_mode_int)
        .rename("target_strat")
        .reset_index()
    )
    return g


def _train_test_stratified_or_shuffle(
    table: pd.DataFrame,
    strat_col: str,
    test_size: float,
    random_state: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    counts = table[strat_col].value_counts()
    if counts.min() >= 2 and len(table) >= 2:
        stratify = table[strat_col]
    try:
        a, b = train_test_split(
            table,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=True,
        )
        if stratify is None:
            print(
                f"[{label}] Stratify skipped (insufficient samples per class); using random group split."
            )
        return a, b
    except ValueError as e:
        print(f"[{label}] Stratify failed ({e}); using random group split.")
        return train_test_split(
            table,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
            shuffle=True,
        )


def create_splits(
    input_path: str,
    output_path: str,
    random_state: int = 42,
    group_col: Optional[str] = None,
    val_fraction: float = 0.15,
    fusion_fraction: float = 0.0,
    holdout_fraction: float = 0.0,
) -> None:
    df = pd.read_csv(input_path)
    print(f"Loaded data shape: {df.shape}")

    resolved = _resolve_group_column(df, group_col)
    if resolved == "lesion_id" and group_col is None:
        print(
            "No patient/source group column found "
            f"(tried {_DEFAULT_GROUP_CANDIDATES}). "
            "Falling back to lesion_id (one lesion = one group). "
            "Add e.g. patient_id to metadata and re-run prepare_data, "
            "or pass --group-col attribution for source-based grouping."
        )
    else:
        print(f"Using group column for splits: {resolved!r}")

    split_key = _split_group_key_series(df, resolved)
    group_df = _group_table_for_stratify(df, split_key)
    print(f"Unique groups ({resolved}, after NA→lesion_id): {len(group_df)}")

    if not 0 < val_fraction < 1 or min(fusion_fraction, holdout_fraction) < 0 or val_fraction + fusion_fraction + holdout_fraction >= 1:
        raise ValueError("Split fractions must be nonnegative, with positive train and validation fractions")
    remaining = group_df
    remaining_fraction = 1.0
    assignments = {}
    for offset, (name, fraction) in enumerate((("val", val_fraction), ("fusion", fusion_fraction), ("holdout", holdout_fraction))):
        if fraction == 0:
            continue
        remaining, selected = _train_test_stratified_or_shuffle(
            remaining, "target_strat", test_size=fraction / remaining_fraction,
            random_state=random_state + offset, label=f"train|{name}")
        assignments[name] = set(selected["_split_group"].astype(str))
        remaining_fraction -= fraction
    sk = split_key.astype(str)
    df["split"] = "train"
    for name, ids in assignments.items():
        df.loc[sk.isin(ids), "split"] = name

    print("Split sizes (rows / images) — training data only:")
    print(df["split"].value_counts().to_string())
    print(
        "\nNote: No held-out 'test' split here. Use MILK10k_Test_Metadata + "
        "MILK10k_Test_Input (prepare_test_data.py) for official test evaluation."
    )

    leakage = df.groupby(split_key.astype(str), observed=False)["split"].nunique()
    leaky = (leakage > 1).sum()
    print("\n--- Leakage check (same group in multiple splits) ---")
    print(f"Groups with >1 split assignment: {leaky}")
    if leaky:
        print("WARNING: leakage detected across groups.")
    else:
        print("No group split leakage. ✔")

    print("\n--- Class counts per split (target) ---")
    for name in ("train", "val", "fusion", "holdout"):
        if name not in set(df["split"]):
            continue
        sub = df.loc[df["split"] == name, "target"]
        print(f"{name}:\n{sub.value_counts().sort_index().to_string()}\n")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved split dataset to: {output_path}")


def _default_paths() -> tuple[str, str]:
    import sys
    from pathlib import Path

    project_src = Path(__file__).resolve().parents[1]
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from components.paths import FINAL_METADATA_CSV, FINAL_METADATA_WITH_SPLIT_CSV

    return str(FINAL_METADATA_CSV), str(FINAL_METADATA_WITH_SPLIT_CSV)


if __name__ == "__main__":
    in_def, out_def = _default_paths()
    p = argparse.ArgumentParser(
        description="Train/val split on training data (no held-out test; use MILK10k_Test_* for test)."
    )
    p.add_argument("--input", default=in_def, help="Input CSV (e.g. final_metadata.csv)")
    p.add_argument("--output", default=out_def, help="Output CSV with split column")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--fusion-fraction", type=float, default=0.0)
    p.add_argument("--holdout-fraction", type=float, default=0.0)
    p.add_argument(
        "--group-col",
        default=None,
        help="Group id column (default: auto patient_id, case_id, subject_id, patient; else lesion_id). "
        "Example for source-based split: --group-col attribution",
    )
    args = p.parse_args()
    create_splits(
        args.input,
        args.output,
        random_state=args.random_state,
        group_col=args.group_col,
        val_fraction=args.val_fraction, fusion_fraction=args.fusion_fraction, holdout_fraction=args.holdout_fraction,
    )
