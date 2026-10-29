from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GradCAMResult:
    cam: np.ndarray  # (H, W) float32 in [0,1]
    logits: torch.Tensor  # (C,)
    pred_class: int


class GradCAM:
    """Minimal Grad-CAM for models with forward(images) -> logits."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        *,
        device: torch.device | str,
        forward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = torch.device(device)
        self.forward_fn = forward_fn

        self._activations: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        self._h1 = target_layer.register_forward_hook(self._forward_hook)
        self._h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        self._h1.remove()
        self._h2.remove()

    def _forward_hook(self, _module, _inp, out):
        self._activations = out

    def _backward_hook(self, _module, _grad_inp, grad_out):
        self._grads = grad_out[0]

    @torch.inference_mode(False)
    def __call__(
        self,
        *,
        images: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> GradCAMResult:
        self.model.eval()
        images = images.to(self.device)
        self.model.zero_grad(set_to_none=True)

        logits = self.forward_fn(images) if self.forward_fn is not None else self.model(images)

        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"Expected logits shape (1, C), got {tuple(logits.shape)}")

        logits_1d = logits[0]
        pred_class = int(torch.argmax(logits_1d).item())
        cls = pred_class if target_class is None else int(target_class)

        score = logits_1d[cls]
        score.backward(retain_graph=False)

        if self._activations is None or self._grads is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        acts = self._activations
        grads = self._grads
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_np = cam.detach().to("cpu").float().numpy().astype(np.float32)
        return GradCAMResult(cam=cam_np, logits=logits_1d.detach(), pred_class=pred_class)
