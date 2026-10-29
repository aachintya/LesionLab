"""Grad-CAM for the existing image branches, with no parameter updates."""
from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def _png_base64(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def grad_cam(model, inputs, image, *, target_class):
    """Explain a branch logit on its exact processed RGB input, not the raw upload.

    autograd.grad reads activation gradients without accumulating parameter .grad
    tensors. Hooks are removed on every path, and evaluation-mode buffers are fixed.
    """
    if model.training:
        raise ValueError("Heatmaps require an image model in evaluation mode.")
    if inputs.ndim != 4 or inputs.shape[0] != 1:
        raise ValueError("Explain exactly one image at a time.")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Heatmap overlays require the processed RGB image.")
    if tuple(inputs.shape[-2:]) != tuple(image.shape[:2]):
        raise ValueError("The heatmap and processed image must share the same coordinate frame.")
    activations = []
    target_layer = model.backbone.features[-1]

    def capture(_module, _args, output):
        # Grad-CAM needs derivatives of the head with respect to this activation,
        # not an input-gradient graph through the frozen feature extractor.
        spatial = output.detach().requires_grad_(True)
        activations.append(spatial)
        return spatial

    hook = target_layer.register_forward_hook(capture)
    try:
        with torch.inference_mode(False), torch.enable_grad():
            explain_input = inputs.detach()
            logits = model(explain_input)
            if not 0 <= target_class < logits.shape[1] or not torch.isfinite(logits).all():
                raise ValueError("The heatmap target or model activations are invalid.")
            if len(activations) != 1 or activations[0].ndim != 4:
                raise ValueError("The image model did not return spatial features.")
            features = activations[0]
            gradient, = torch.autograd.grad(logits[0, target_class], features)
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            heat = (weights * features).sum(dim=1, keepdim=True).relu()
            heat = F.interpolate(heat, size=image.shape[:2], mode="bilinear", align_corners=False)[0, 0]
            heat = heat.detach().float().cpu().numpy()
        if not np.isfinite(heat).all():
            raise ValueError("The image model produced a nonfinite explanation.")
        low, high = float(heat.min()), float(heat.max())
        if high - low <= 1e-12:
            return {"status": "unavailable", "message": "No spatially varying positive Grad-CAM signal was available for this category."}
        heat = (heat - low) / (high - low)
        color = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.rint(heat * 255)), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        # Low-activation pixels retain the processed photograph's appearance.
        opacity = (0.58 * heat[..., None]).astype(np.float32)
        overlay = np.uint8(np.clip(np.rint(image * (1 - opacity) + color * opacity), 0, 255))
        return {"status": "available", "method": "Grad-CAM", "target_index": int(target_class),
                "coordinate_frame": "processed_model_input", "input_size": [image.shape[1], image.shape[0]],
                "input_png_base64": _png_base64(image), "overlay_png_base64": _png_base64(overlay),
                "message": "Relative spatial influence on this image branch's target-class score; not a lesion boundary or a fusion-model explanation."}
    finally:
        hook.remove()
