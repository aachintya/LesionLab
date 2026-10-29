"""Encode lesion-level tabular metadata for the metadata MLP model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_NUMERIC_COLS = ("age_approx", "skin_tone_class")
DEFAULT_CATEGORICAL_COLS = ("sex", "site")


class MetadataEncoder:
    """Fit on train split; transform numeric + categorical lesion metadata."""

    def __init__(
        self,
        numeric_cols: tuple[str, ...] = DEFAULT_NUMERIC_COLS,
        categorical_cols: tuple[str, ...] = DEFAULT_CATEGORICAL_COLS,
    ):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.numeric_means_: dict[str, float] = {}
        self.numeric_stds_: dict[str, float] = {}
        self.category_maps_: dict[str, dict[str, int]] = {}
        self.feature_names_: list[str] = []
        self.input_dim_: int = 0

    def _available_cols(self, df: pd.DataFrame, cols: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(c for c in cols if c in df.columns)

    def fit(self, df: pd.DataFrame) -> "MetadataEncoder":
        numeric_cols = self._available_cols(df, self.numeric_cols)
        categorical_cols = self._available_cols(df, self.categorical_cols)

        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            mean = float(values.mean())
            if not np.isfinite(mean):
                mean = 0.0
            std = float(values.std())
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            self.numeric_means_[col] = mean
            self.numeric_stds_[col] = std

        for col in categorical_cols:
            series = df[col].astype("string").fillna("unknown")
            categories = sorted(series.unique())
            self.category_maps_[col] = {cat: idx for idx, cat in enumerate(categories)}

        self.feature_names_ = []
        for col in numeric_cols:
            self.feature_names_.append(col)
        for col in categorical_cols:
            for cat in self.category_maps_[col]:
                self.feature_names_.append(f"{col}={cat}")

        self.input_dim_ = len(numeric_cols) + sum(
            len(self.category_maps_[col]) for col in categorical_cols
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.feature_names_:
            raise RuntimeError("MetadataEncoder must be fit before transform.")

        numeric_cols = tuple(self.numeric_means_.keys())
        categorical_cols = tuple(self.category_maps_.keys())
        rows: list[list[float]] = []

        for _, row in df.iterrows():
            feats: list[float] = []
            for col in numeric_cols:
                val = pd.to_numeric(row.get(col), errors="coerce")
                if not np.isfinite(val):
                    val = self.numeric_means_[col]
                feats.append((float(val) - self.numeric_means_[col]) / self.numeric_stds_[col])

            for col in categorical_cols:
                raw = row.get(col)
                key = "unknown" if pd.isna(raw) else str(raw)
                one_hot = [0.0] * len(self.category_maps_[col])
                idx = self.category_maps_[col].get(key)
                if idx is None:
                    idx = self.category_maps_[col].get("unknown")
                if idx is not None:
                    one_hot[idx] = 1.0
                feats.extend(one_hot)

            rows.append(feats)

        return np.asarray(rows, dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        payload: dict[str, Any] = {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_means": self.numeric_means_,
            "numeric_stds": self.numeric_stds_,
            "category_maps": self.category_maps_,
            "feature_names": self.feature_names_,
            "input_dim": self.input_dim_,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MetadataEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        enc = cls(
            numeric_cols=tuple(payload["numeric_cols"]),
            categorical_cols=tuple(payload["categorical_cols"]),
        )
        enc.numeric_means_ = {k: float(v) for k, v in payload["numeric_means"].items()}
        enc.numeric_stds_ = {k: float(v) for k, v in payload["numeric_stds"].items()}
        enc.category_maps_ = payload["category_maps"]
        enc.feature_names_ = payload["feature_names"]
        enc.input_dim_ = int(payload["input_dim"])
        return enc
