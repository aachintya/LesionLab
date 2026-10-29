"""EfficientNetV2-S image-only classifier."""

from __future__ import annotations

# Source foundation: asyagenc/skin-cancer-detection- (upstream contributors).
# https://github.com/asyagenc/skin-cancer-detection-
# Starting checkout: 4787606a2c04ee7a3c2542994c4ff3b5ac509daa.
# LesionLab extends the image/metadata/fusion pipeline with separate research work.

import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


def configure_torchvision_pretrained_environment(*, cwd: Path | None = None) -> None:
    base = cwd if cwd is not None else Path.cwd()
    torch_home = str(base.resolve() / ".torch")
    os.environ.setdefault("TORCH_HOME", torch_home)


def load_efficientnet_v2_s_backbone(*, pretrained: bool = True) -> nn.Module:
    configure_torchvision_pretrained_environment()
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    return efficientnet_v2_s(weights=weights)


class EfficientNetV2Image(nn.Module):
    """EfficientNetV2-S feature extractor + separate classification head."""

    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.backbone = backbone

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.head(features)


def build_image_model(num_classes: int, dropout: float = 0.4, *, pretrained: bool = True) -> EfficientNetV2Image:
    backbone = load_efficientnet_v2_s_backbone(pretrained=pretrained)
    return EfficientNetV2Image(backbone=backbone, num_classes=num_classes, dropout=dropout)
