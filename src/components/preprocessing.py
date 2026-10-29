"""Skin lesion preprocessing: hair removal (black-hat + inpaint), CLAHE, resize."""

from __future__ import annotations

import cv2
import numpy as np

# Hair removal (black-hat)
HAIR_KERNEL_SIZE = (21, 21)
BLACKHAT_THRESHOLD = 20

# Inpainting
INPAINT_RADIUS = 3

# CLAHE (LAB L-channel)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Final smoothing / geometry
MEDIAN_BLUR_KSIZE = 3
DEFAULT_TARGET_SIZE = (224, 224)


def create_hair_mask(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (blackhat visualization, binary hair mask uint8)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, HAIR_KERNEL_SIZE)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, BLACKHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    return blackhat, mask


def apply_inpainting(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(img, mask, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_final_steps(img: np.ndarray, target_size: tuple[int, int] = DEFAULT_TARGET_SIZE) -> np.ndarray:
    filtered = cv2.medianBlur(img, MEDIAN_BLUR_KSIZE)
    return cv2.resize(filtered, target_size, interpolation=cv2.INTER_CUBIC)


def preprocess_for_modality(img_rgb: np.ndarray, modality: str) -> np.ndarray:
    """
    Modality-specific OpenCV preprocessing before torchvision augmentations.

    - dermoscopic: hair removal + CLAHE (fine structures, hair artifacts)
    - clinical: CLAHE only (broader scene, less hair artifact)
    """
    if modality == "dermoscopic":
        _, hair_mask = create_hair_mask(img_rgb)
        img_rgb = apply_inpainting(img_rgb, hair_mask)
        return apply_clahe(img_rgb)
    if modality == "clinical":
        return apply_clahe(img_rgb)
    raise ValueError(f"Unknown modality {modality!r}. Expected 'clinical' or 'dermoscopic'.")


def extract_lesion_roi_foreground(
    img_rgb: np.ndarray,
    *,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.95,
    pad_ratio: float = 0.05,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """
    Heuristic ROI crop using LAB 'a' channel + Otsu (no GT masks).

    Returns cropped RGB (or original if no plausible ROI) and bbox (x1,y1,x2,y2) or None.
    """
    h, w = img_rgb.shape[:2]
    img_area = float(h * w)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    _, a, _ = cv2.split(lab)

    a_blur = cv2.GaussianBlur(a, (5, 5), 0)
    _, mask = cv2.threshold(a_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    area_fg = float((mask > 0).sum())
    if not (min_area_ratio * img_area <= area_fg <= max_area_ratio * img_area):
        mask = cv2.bitwise_not(mask)

    k = max(3, int(round(min(h, w) * 0.01)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb, None

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if not (min_area_ratio * img_area <= area <= max_area_ratio * img_area):
        return img_rgb, None

    x, y, bw, bh = cv2.boundingRect(cnt)
    pad_x = int(round(pad_ratio * w))
    pad_y = int(round(pad_ratio * h))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w - 1, x + bw + pad_x)
    y2 = min(h - 1, y + bh + pad_y)

    roi = img_rgb[y1 : y2 + 1, x1 : x2 + 1]
    return roi, (x1, y1, x2, y2)
