"""Verify actual gradient localization, coordinate alignment, and read-only state."""
import base64
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.image_explanations import grad_cam


class TinyImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 1, 1, bias=False))
        with torch.no_grad():self.features[0].weight.fill_(1 / 3)
        self.backbone = SimpleNamespace(features=self.features)

    def forward(self, images):
        signal = self.features(images).mean((2, 3))
        return torch.cat([signal, -signal], dim=1)


class HeatmapTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyImageModel().eval()
        self.inputs = torch.zeros(1, 3, 12, 12)
        self.inputs[:, :, :4, :4] = 1
        self.image = np.uint8(self.inputs[0].permute(1, 2, 0).numpy() * 255)

    def test_real_gradient_highlights_the_causal_patch_without_updating_state(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        expected = self.model(self.inputs).detach().clone()
        result = grad_cam(self.model, self.inputs, self.image, target_class=0)
        self.assertEqual(result["status"], "available")
        original = np.array(Image.open(BytesIO(base64.b64decode(result["input_png_base64"]))))
        overlay = np.array(Image.open(BytesIO(base64.b64decode(result["overlay_png_base64"]))))
        np.testing.assert_array_equal(original, self.image)
        np.testing.assert_array_equal(overlay[5:, 5:], self.image[5:, 5:])
        self.assertFalse(np.array_equal(overlay[:4, :4], self.image[:4, :4]))
        for k, v in self.model.state_dict().items():torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        torch.testing.assert_close(self.model(self.inputs), expected, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertFalse(self.model.training)
        self.assertFalse(self.model.features[-1]._forward_hooks)

    def test_no_positive_signal_is_unavailable_instead_of_a_fabricated_map(self):
        result = grad_cam(self.model, self.inputs, self.image, target_class=1)
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("overlay_png_base64", result)
        self.assertFalse(self.model.features[-1]._forward_hooks)

    def test_frozen_backbone_needs_no_input_gradient_graph(self):
        self.model.requires_grad_(False)
        requires_grad = []
        hook = self.model.features[0].register_forward_pre_hook(
            lambda _module, args: requires_grad.append(args[0].requires_grad))
        try:
            result = grad_cam(self.model, self.inputs, self.image, target_class=0)
        finally:
            hook.remove()
        self.assertEqual(result["status"], "available")
        self.assertEqual(requires_grad, [False])
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_errors_remove_hooks_and_reject_misaligned_images(self):
        with self.assertRaises(ValueError):grad_cam(self.model, self.inputs, self.image, target_class=99)
        self.assertFalse(self.model.features[-1]._forward_hooks)
        with self.assertRaises(ValueError):grad_cam(self.model, self.inputs, self.image[:5], target_class=0)
        self.model.train()
        with self.assertRaises(ValueError):grad_cam(self.model, self.inputs, self.image, target_class=0)
        self.assertTrue(self.model.training)


if __name__ == "__main__":unittest.main()
