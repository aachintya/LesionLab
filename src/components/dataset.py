from __future__ import annotations

import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import (
    DEFAULT_TARGET_SIZE,
)

try:
    from components.metadata_encoder import MetadataEncoder
except ModuleNotFoundError:
    from src.components.metadata_encoder import MetadataEncoder  # type: ignore

if TYPE_CHECKING:
    import pandas as pd


class MILK10kDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        transform=None,
        mask_dir: str | None = None,
        mask_mode: str = "none",
        mask_ext: str = "png",
        roi_mode: str = "none",
        modality: str = "dermoscopic",
        strong_transform=None,
        strong_targets=None,
        cache_dir: str | None = None,
    ):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.mask_dir = mask_dir
        self.mask_mode = mask_mode
        self.mask_ext = mask_ext
        self.roi_mode = roi_mode
        self.modality = modality
        self.cache_dir = cache_dir
        # Class-aware augmentation: samples whose target is in strong_targets
        # get the heavier strong_transform instead of the default transform.
        self.strong_transform = strong_transform
        self.strong_targets = set(strong_targets) if strong_targets else set()

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, idx: int) -> tuple[str, str]:
        img_id = self.df.iloc[idx]["isic_id"]
        img_id_str = str(img_id)
        lesion_id = None
        if "lesion_id" in self.df.columns:
            lesion_id = self.df.iloc[idx]["lesion_id"]

        if lesion_id:
            img_path = os.path.join(self.image_dir, str(lesion_id), f"{img_id_str}.jpg")
        else:
            img_path = os.path.join(self.image_dir, f"{img_id_str}.jpg")

        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f"{img_id_str}.jpg")

        return img_path, img_id_str

    def _load_mask(self, img_id: str, target_size: tuple[int, int]) -> np.ndarray | None:
        if not self.mask_dir:
            return None

        mask_path = os.path.join(self.mask_dir, f"{img_id}.{self.mask_ext}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        tw, th = target_size
        if mask.shape[:2] != (th, tw):
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = (mask > 127).astype(np.uint8) * 255

        return mask

    def _apply_mask_mode(self, image_rgb: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
        if self.mask_mode == "none":
            return image_rgb

        if self.mask_mode == "masked":
            mask3 = cv2.merge([lesion_mask, lesion_mask, lesion_mask])
            return cv2.bitwise_and(image_rgb, mask3)

        if self.mask_mode == "crop":
            ys, xs = (lesion_mask > 0).nonzero()
            if len(xs) == 0 or len(ys) == 0:
                return image_rgb

            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            h, w = lesion_mask.shape[:2]
            pad_x = max(1, int(0.05 * w))
            pad_y = max(1, int(0.05 * h))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w - 1, x2 + pad_x)
            y2 = min(h - 1, y2 + pad_y)

            cropped = image_rgb[y1 : y2 + 1, x1 : x2 + 1]
            return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

        raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

    def __getitem__(self, idx: int):
        img_path, img_id = self._resolve_image_path(idx)
        from components.image_cache import load_preprocessed_image
        image = load_preprocessed_image(img_path, modality=self.modality,
                                        roi_mode=self.roi_mode, cache_dir=self.cache_dir)

        lesion_mask = self._load_mask(img_id, target_size=DEFAULT_TARGET_SIZE)
        if lesion_mask is not None:
            image = self._apply_mask_mode(image, lesion_mask)

        image = Image.fromarray(image)

        target = int(self.df.iloc[idx]["target"]) if "target" in self.df.columns else None
        transform = self.transform
        if (
            self.strong_transform is not None
            and target is not None
            and target in self.strong_targets
        ):
            transform = self.strong_transform
        if transform:
            image = transform(image)

        label = torch.tensor(target if target is not None else 0, dtype=torch.long)
        return image, label


class LesionMetadataDataset(Dataset):
    """Lesion-level tabular metadata (one row per lesion_id)."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        encoder: MetadataEncoder,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.encoder = encoder
        self.features = encoder.transform(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        if "target" in self.df.columns:
            label = torch.tensor(int(self.df.iloc[idx]["target"]), dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)
        lesion_id = str(self.df.iloc[idx]["lesion_id"])
        return features, label, lesion_id
