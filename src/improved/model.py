"""Joint feature fusion, initialized from the frozen v1 image encoders."""
from pathlib import Path

import torch
from torch import nn

from components.paths import NUM_CLASSES
from models.efficientnet_v2_image import build_image_model


class PairedLesionModel(nn.Module):
    def __init__(self, metadata_dim, *, width=128, branches=None):
        super().__init__()
        self.clinical, self.dermoscopic = (branches if branches is not None else
                                         (build_image_model(NUM_CLASSES, pretrained=False),
                                          build_image_model(NUM_CLASSES, pretrained=False)))
        feature_dim = self.clinical.head[-1].in_features
        self.projections = nn.ModuleList([nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width),
                                                        nn.GELU()) for _ in range(2)])
        self.metadata = nn.Sequential(nn.Linear(metadata_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2))
        self.gate = nn.Linear(width * 2 + 64, 2)
        self.fusion = nn.Sequential(nn.LayerNorm(width * 4 + 64), nn.Linear(width * 4 + 64, 128),
                                    nn.GELU(), nn.Dropout(0.35), nn.Linear(128, NUM_CLASSES))
        self.metadata_dropout = 0.15

    def warm_start(self, checkpoint_dir):
        for view in ("clinical", "dermoscopic"):
            state = torch.load(Path(checkpoint_dir) / f"best_{view}.pt", map_location="cpu", weights_only=True)
            getattr(self, view).load_state_dict(state, strict=True)

    def freeze_encoders(self, frozen):
        for branch in (self.clinical, self.dermoscopic):
            for param in branch.backbone.parameters():
                param.requires_grad = not frozen

    def train(self, mode=True):
        super().train(mode)
        # Small batches: keep the pretrained BN statistics even during fine tuning.
        for branch in (self.clinical, self.dermoscopic):
            for layer in branch.backbone.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
            if not any(p.requires_grad for p in branch.backbone.parameters()):
                branch.backbone.eval()
        return self

    def forward(self, clinical, dermoscopic, metadata):
        c_raw = self.clinical.backbone(clinical)
        d_raw = self.dermoscopic.backbone(dermoscopic)
        c, d = self.projections[0](c_raw), self.projections[1](d_raw)
        if self.training and self.metadata_dropout:
            metadata = metadata * (torch.rand((len(metadata), 1), device=metadata.device) >= self.metadata_dropout)
        m = self.metadata(metadata)
        weights = self.gate(torch.cat([c, d, m], dim=1)).softmax(dim=1)
        c, d = c * weights[:, :1], d * weights[:, 1:]
        auxiliary_c, auxiliary_d = self.clinical.head(c_raw), self.dermoscopic.head(d_raw)
        combined = torch.cat([c, d, torch.abs(c - d), c * d, m], dim=1)
        logits = self.fusion(combined) + 0.5 * (auxiliary_c + auxiliary_d)
        return {"logits": logits, "clinical_logits": auxiliary_c, "dermoscopic_logits": auxiliary_d}


def balanced_loss(outputs, targets, counts, *, auxiliary=0.2):
    # Natural sampling + Balanced Softmax; do not add an inverse-frequency sampler.
    log_counts = counts.to(outputs["logits"].device).float().log()
    criterion = nn.CrossEntropyLoss()
    main = criterion(outputs["logits"].float() + log_counts, targets)
    aux = sum(criterion(outputs[key].float() + log_counts, targets)
              for key in ("clinical_logits", "dermoscopic_logits")) / 2
    return main + auxiliary * aux
