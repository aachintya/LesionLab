"""PyTorch MLP for lesion-level tabular metadata."""

from __future__ import annotations

import torch
import torch.nn as nn

METADATA_CHECKPOINT_NAME = "best_metadata.pt"
METADATA_ENCODER_NAME = "metadata_encoder.json"


class MetadataMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def build_metadata_model(
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> MetadataMLP:
    return MetadataMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
