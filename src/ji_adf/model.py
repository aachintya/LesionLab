"""JI-ADF core: arXiv:2604.27343v2, equations 3–11.

The paired-image embedding and unspecified layer widths are explicit project
choices. This is a documented adaptation, not a claim of exact reproduction.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_v2_s

from components.paths import NUM_CLASSES

PAPER_URL = "https://arxiv.org/html/2604.27343v2"
VARIANTS = ("ji_adf", "concat_fixed", "mmfa_fixed", "concat_adaptive")
BRANCH_ORDER = ("image", "joint", "metadata")
PRECISION_POLICY = "FP32 encoders, projections, attention, heads and loss; T4 FP16 overflow fix"


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights=None)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

    def forward(self, image):
        # The T4 run reproduced finite inputs/weights but nonfinite FP16 encoder
        # activations. The same weights and input remained finite in FP32.
        with torch.autocast(image.device.type, enabled=False):
            return self.backbone(image.float())


class MMFA(nn.Module):
    """Modality-specific Q/K/V, two-token attention and residual (eqs. 8–11)."""
    def __init__(self, image_dim, metadata_dim, joint_dim=512, heads=4, head_dim=64):
        super().__init__()
        if min(image_dim, metadata_dim, joint_dim, heads, head_dim) < 1:
            raise ValueError("MMFA dimensions must be positive")
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.image_qkv = nn.ModuleDict({key: nn.Linear(image_dim, inner, bias=False) for key in ("q", "k", "v")})
        self.metadata_qkv = nn.ModuleDict({key: nn.Linear(metadata_dim, inner, bias=False) for key in ("q", "k", "v")})
        self.output = nn.Linear(2 * inner, joint_dim, bias=False)
        self.transform = nn.Sequential(nn.Linear(joint_dim, joint_dim), nn.ReLU())
        self.skip = nn.Linear(image_dim + metadata_dim, joint_dim, bias=False)

    def forward(self, image_features, metadata_features):
        # QK^T can overflow FP16 before softmax, even when Q/K are finite.
        # Enforce FP32 even when an external caller has enabled autocast.
        with torch.autocast(image_features.device.type, enabled=False):
            return self._forward_float32(image_features.float(), metadata_features.float())

    def _forward_float32(self, image_features, metadata_features):
        batch = len(image_features)
        tokens = {}
        for key in ("q", "k", "v"):
            visual = self.image_qkv[key](image_features).reshape(batch, self.heads, self.head_dim)
            metadata = self.metadata_qkv[key](metadata_features).reshape(batch, self.heads, self.head_dim)
            tokens[key] = torch.stack([visual, metadata], dim=2)
        attention = (tokens["q"] @ tokens["k"].transpose(-1, -2) / math.sqrt(self.head_dim)).softmax(dim=-1)
        attended = (attention @ tokens["v"]).flatten(1)
        return self.skip(torch.cat([image_features, metadata_features], dim=1)) + self.transform(self.output(attended))


def fuse_decisions(branch_logits, weight_logits=None):
    """Convex probability mixture, evaluated stably in log space (eqs. 5–6).

    Applying CE directly to probabilities would apply softmax twice. The caller
    uses NLL on the normalized log mixture instead.
    """
    if branch_logits.ndim != 3 or branch_logits.shape[1] != 3:
        raise ValueError("Expected batch x three branches x classes")
    log_probs = F.log_softmax(branch_logits.float(), dim=-1)
    log_weights = (F.log_softmax(weight_logits.float(), dim=1) if weight_logits is not None else
                   log_probs.new_full((len(log_probs), 3), -math.log(3)))
    mixed = torch.logsumexp(log_probs + log_weights.unsqueeze(-1), dim=1)
    return mixed, log_weights.exp()


class JIADFModel(nn.Module):
    def __init__(self, metadata_dim, *, variant="ji_adf", image_dim=512, metadata_width=256,
                 joint_dim=512, heads=4, head_dim=64, encoders=None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown JI-ADF variant: {variant}")
        self.variant = variant
        self.clinical, self.dermoscopic = encoders if encoders is not None else (ImageEncoder(), ImageEncoder())
        visual_dim = self.clinical.feature_dim + self.dermoscopic.feature_dim
        self.image_projection = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, image_dim),
                                               nn.ReLU(), nn.Dropout(0.2))
        self.metadata_encoder = nn.Sequential(nn.Linear(metadata_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                                               nn.Linear(128, metadata_width), nn.ReLU())
        # Shared modules are constructed before the ablation-specific fusion module.
        self.image_head = nn.Linear(image_dim, NUM_CLASSES)
        self.metadata_head = nn.Linear(metadata_width, NUM_CLASSES)
        self.joint_head = nn.Linear(joint_dim, NUM_CLASSES)
        self.decision_gate = nn.Sequential(nn.Linear(NUM_CLASSES * 3, 64), nn.ReLU(), nn.Linear(64, 3))
        self.joint_fusion = (MMFA(image_dim, metadata_width, joint_dim, heads, head_dim)
                             if variant in ("ji_adf", "mmfa_fixed") else
                             nn.Sequential(nn.Linear(image_dim + metadata_width, joint_dim), nn.ReLU()))

    def initialize(self, reference, initialization="imagenet"):
        reference = Path(reference)
        if initialization == "imagenet":
            state = torch.load(reference / "imagenet_efficientnet_v2_s.pt", map_location="cpu", weights_only=True)
            features = {key: value for key, value in state.items() if not key.startswith("classifier.")}
            for branch in (self.clinical, self.dermoscopic):
                branch.backbone.load_state_dict(features, strict=True)
        elif initialization == "v1":
            for name in ("clinical", "dermoscopic"):
                state = torch.load(reference / "checkpoints" / f"best_{name}.pt", map_location="cpu", weights_only=True)
                features = {key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")}
                getattr(self, name).backbone.load_state_dict(features, strict=True)
        else:
            raise ValueError("Initialization must be imagenet or v1")

    def freeze_encoders(self, frozen):
        for branch in (self.clinical, self.dermoscopic):
            for parameter in branch.parameters():
                parameter.requires_grad = not frozen

    def train(self, mode=True):
        super().train(mode)
        # T4 microbatches are smaller than the paper's batch. Freeze BN running
        # statistics; affine parameters still learn. This adaptation is recorded.
        for branch in (self.clinical, self.dermoscopic):
            if not any(p.requires_grad for p in branch.parameters()):
                branch.eval()
            for layer in branch.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        return self

    def forward(self, clinical, dermoscopic, metadata):
        features = torch.cat([self.clinical(clinical), self.dermoscopic(dermoscopic)], dim=1)
        with torch.autocast(features.device.type, enabled=False):
            visual = self.image_projection(features.float())
            context = self.metadata_encoder(metadata.float())
            joint = (self.joint_fusion(visual, context) if isinstance(self.joint_fusion, MMFA) else
                     self.joint_fusion(torch.cat([visual, context], dim=1)))
            branch_logits = torch.stack([self.image_head(visual), self.joint_head(joint), self.metadata_head(context)], dim=1)
            adaptive = self.variant in ("ji_adf", "concat_adaptive")
            weight_logits = self.decision_gate(branch_logits.flatten(1)) if adaptive else None
            log_probs, weights = fuse_decisions(branch_logits, weight_logits)
        return {"log_probs": log_probs, "branch_logits": branch_logits, "branch_weights": weights}


def ji_adf_loss(outputs, targets):
    """Equation 7: final NLL + .5 joint CE + .25 image CE + .25 metadata CE."""
    logits = outputs["branch_logits"].float()
    final = F.nll_loss(outputs["log_probs"].float(), targets)
    image = F.cross_entropy(logits[:, 0], targets)
    joint = F.cross_entropy(logits[:, 1], targets)
    metadata = F.cross_entropy(logits[:, 2], targets)
    return final + 0.25 * image + 0.5 * joint + 0.25 * metadata
