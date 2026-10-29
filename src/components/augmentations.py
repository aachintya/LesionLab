"""Modality-specific train/val transforms for the 3-model ensemble pipeline."""

from __future__ import annotations

from torchvision import transforms

from components.data_utils import MODALITY_IMAGE_TYPES

# ImageNet normalization (must match training / val / inference)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Mixup strength per image modality (metadata model does not use mixup)
MIXUP_ALPHA = {
    "clinical": 0.4,
    "dermoscopic": 0.25,
}


def _normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _val_test_transform() -> transforms.Compose:
    """Deterministic post-preprocessing pipeline (images already resized to 224)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            _normalize(),
        ]
    )


def _clinical_train_transform() -> transforms.Compose:
    """Clinical close-up: geometric augmentation only."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            _normalize(),
        ]
    )


def _dermoscopic_train_transform() -> transforms.Compose:
    """Dermoscopic: light geometric augmentation only."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            _normalize(),
        ]
    )


def _clinical_strong_transform() -> transforms.Compose:
    """Heavier (but lesion-safe) augmentation for rare clinical classes.

    Conservative crop keeps the lesion in frame; mild color jitter preserves
    diagnostic color; cutout (RandomErasing) adds occlusion robustness.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            _normalize(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
        ]
    )


def _dermoscopic_strong_transform() -> transforms.Compose:
    """Heavier augmentation for rare dermoscopic classes.

    Extra-conservative: dermoscopy color/vascular patterns are diagnostic, so
    crop scale and color jitter are kept small.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.11)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            _normalize(),
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.08)),
        ]
    )


def get_strong_transform(*, modality: str) -> transforms.Compose:
    """Stronger train transform used for rare classes only (class-aware aug)."""
    if modality == "clinical":
        return _clinical_strong_transform()
    if modality == "dermoscopic":
        return _dermoscopic_strong_transform()
    raise ValueError(f"No strong transform for modality {modality!r}.")


def get_transforms(*, modality: str) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Return (train_transform, val_test_transform) for an image modality.

    Parameters
    ----------
    modality:
        ``"clinical"`` or ``"dermoscopic"``. Metadata models do not use image transforms.
    """
    if modality not in MODALITY_IMAGE_TYPES:
        raise ValueError(
            f"Unknown image modality {modality!r}. Expected one of: {list(MODALITY_IMAGE_TYPES)}"
        )

    val_test = _val_test_transform()
    if modality == "clinical":
        return _clinical_train_transform(), val_test
    return _dermoscopic_train_transform(), val_test


def get_mixup_alpha(modality: str) -> float:
    if modality not in MIXUP_ALPHA:
        raise ValueError(f"Mixup is not defined for modality {modality!r}.")
    return MIXUP_ALPHA[modality]
