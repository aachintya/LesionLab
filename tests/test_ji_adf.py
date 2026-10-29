import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ji_adf.model import ImageEncoder, JIADFModel, MMFA, fuse_decisions, ji_adf_loss


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 4
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, image):
        return self.backbone(image)


class TinyJIADF(JIADFModel):
    def __init__(self, metadata_dim, *, variant="ji_adf"):
        super().__init__(metadata_dim, variant=variant, image_dim=8, metadata_width=6, joint_dim=8,
                         heads=2, head_dim=2, encoders=(TinyEncoder(), TinyEncoder()))

    def initialize(self, _reference, _initialization="imagenet"):
        pass


class JIADFTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(2)

    def fixture(self):
        from utils.baselines import sha256_file
        from utils.evaluation_protocol import metric_protocol
        reference, images = self.root / "reference", self.root / "images"
        reference.mkdir()
        rows = []
        for split in ("train", "val", "fusion", "holdout"):
            for label in range(11):
                lesion = f"{split}_{label}"
                for view, kind in (("clinical", "clinical: close-up"), ("dermoscopic", "dermoscopic")):
                    identity = f"{lesion}_{view}"
                    path = images / lesion / f"{identity}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (90, 70), (30 + label * 10, 80, 90)).save(path)
                    rows.append(dict(lesion_id=lesion, isic_id=identity, image_type=kind, split=split,
                                     target=label, age_approx=40 if split == "train" else 900,
                                     sex="female" if split == "train" else "unknown", site="arm", skin_tone_class=2))
        pd.DataFrame(rows).sample(frac=1, random_state=13).to_csv(reference / "split.csv", index=False)
        (reference / "checkpoints").mkdir()
        (reference / "checkpoints/input_validator.pt").write_bytes(b"fixture gate")
        manifest = {"sha256": {name: sha256_file(reference / name) for name in ("split.csv", "checkpoints/input_validator.pt")},
                    "raw_release_sha256": {}}
        (reference / "manifest.json").write_text(json.dumps(manifest))
        report = {"evaluation_split": "holdout", "evaluation_lesions": 11, "tta": False, "upload_gate_enabled": False,
                  "accuracy": 0.1, "balanced_accuracy": 0.1, "macro_f1": 0.1, "csv_sha256": sha256_file(reference / "split.csv"),
                  **metric_protocol(reference / "split.csv", "holdout")}
        (reference / "metrics.json").write_text(json.dumps(report))
        return reference, images

    def args(self, reference, images, variant="ji_adf", parent="run"):
        return argparse.Namespace(reference=reference, images=images, output=self.root / parent / variant / "seed-42",
                                  cache=None, variant=variant, initialization="imagenet", seed=42, epochs=3,
                                  freeze_epochs=0, patience=5, size=64, batch_size=4, accumulate=2, head_lr=1e-3,
                                  backbone_lr=1e-4, weight_decay=1e-5, workers=0, cpu_threads=2, device="cpu", resume=True)

    def test_mmfa_matches_known_uniform_attention_and_residual(self):
        layer = MMFA(2, 2, joint_dim=2, heads=1, head_dim=2)
        with torch.no_grad():
            for stream in (layer.image_qkv, layer.metadata_qkv):
                stream["q"].weight.zero_()
                stream["k"].weight.zero_()
                stream["v"].weight.copy_(torch.eye(2))
            layer.output.weight.copy_(torch.cat([torch.eye(2), torch.zeros(2, 2)], dim=1))
            layer.transform[0].weight.copy_(torch.eye(2))
            layer.transform[0].bias.zero_()
            layer.skip.weight.zero_()
        image, metadata = torch.tensor([[2., 4.]]), torch.tensor([[6., 8.]])
        torch.testing.assert_close(layer(image, metadata), torch.tensor([[4., 6.]]))
        with torch.no_grad():
            layer.transform[0].weight.zero_()
            layer.skip.weight.copy_(torch.cat([torch.eye(2), torch.eye(2)], dim=1))
        torch.testing.assert_close(layer(image, metadata), torch.tensor([[8., 12.]]))

    def test_probability_mixture_is_convex_and_extreme_logits_are_stable(self):
        logits = torch.tensor([[[2., 0.], [0., 2.], [0., 0.]]], requires_grad=True)
        gates = torch.tensor([[0., np.log(2), 0.]], dtype=torch.float32, requires_grad=True)
        log_prob, weights = fuse_decisions(logits, gates)
        torch.testing.assert_close(weights, torch.tensor([[0.25, 0.5, 0.25]]))
        p = 1 / (1 + np.exp(-2))
        torch.testing.assert_close(log_prob.exp()[0, 0], torch.tensor(0.25 * p + 0.5 * (1-p) + 0.125, dtype=torch.float32))
        log_prob.neg().sum().backward()
        self.assertIsNotNone(gates.grad)
        extreme = torch.tensor([[[1e4, -1e4], [-1e4, 1e4], [1e4, -1e4]]])
        result, _ = fuse_decisions(extreme, torch.tensor([[1e4, -1e4, 0.]]))
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result.exp().sum(1), torch.ones(1))

    def test_attention_avoids_fp16_overflow_and_keeps_finite_gradients(self):
        layer = MMFA(4, 4, joint_dim=4, heads=1, head_dim=4)
        with torch.no_grad():
            for stream in (layer.image_qkv, layer.metadata_qkv):
                for name in stream:
                    stream[name].weight.copy_(torch.eye(4))
        image = torch.full((2, 4), 300.0, requires_grad=True)
        metadata = image.detach().clone().requires_grad_()
        # The original QK matmul exceeds FP16's 65504 limit before softmax.
        with torch.autocast("cpu", dtype=torch.float16):
            query = torch.full((1, 1, 2, 4), 300.0)
            unsafe = query @ query.transpose(-1, -2)
            self.assertFalse(torch.isfinite(unsafe).all())
            actual = layer(image, metadata)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, layer(image, metadata))
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in layer.parameters()))

    def test_all_fusion_heads_stay_fp32_under_outer_autocast(self):
        for variant in ("ji_adf", "concat_fixed", "mmfa_fixed", "concat_adaptive"):
            model = TinyJIADF(5, variant=variant)
            images = torch.randn(2, 3, 16, 16)
            with torch.autocast("cpu", dtype=torch.float16):
                output = model(images, images, torch.randn(2, 5))
                loss = ji_adf_loss(output, torch.tensor([0, 1]))
            self.assertTrue(all(value.dtype == torch.float32 for value in output.values()))
            self.assertTrue(torch.isfinite(loss))

    def test_image_encoder_stays_fp32_even_under_external_autocast(self):
        class OverflowBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Sequential(nn.Identity(), nn.Linear(4, 11))
                self.features = nn.Linear(4, 4, bias=False)
                nn.init.constant_(self.features.weight, 300.0)

            def forward(self, image):
                return self.features(image)

        with patch("ji_adf.model.efficientnet_v2_s", return_value=OverflowBackbone()):
            encoder = ImageEncoder()
        values = torch.full((2, 4), 300.0)
        with torch.autocast("cpu", dtype=torch.float16):
            self.assertFalse(torch.isfinite(encoder.backbone(values)).all())
            actual = encoder(values)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        actual.mean().backward()
        self.assertTrue(torch.isfinite(encoder.backbone.features.weight.grad).all())

    def test_objective_uses_nll_and_paper_auxiliary_weights(self):
        branch = torch.zeros(1, 3, 11)
        branch[0, :, 0] = torch.tensor([1., 2., 3.])
        log_prob, weights = fuse_decisions(branch)
        output = {"log_probs": log_prob, "branch_logits": branch, "branch_weights": weights}
        target_probability = np.exp([1., 2., 3.]) / (np.exp([1., 2., 3.]) + 10)
        expected = -np.log(target_probability.mean()) - np.dot([0.25, 0.5, 0.25], np.log(target_probability))
        self.assertAlmostEqual(ji_adf_loss(output, torch.tensor([0])).item(), expected, places=6)

    def test_all_variants_normalize_and_full_model_learns_every_branch(self):
        for variant in ("ji_adf", "concat_fixed", "mmfa_fixed", "concat_adaptive"):
            model = TinyJIADF(5, variant=variant).train()
            before = model.clinical.backbone[1].running_mean.clone()
            images = torch.randn(3, 3, 16, 16)
            output = model(images, images.flip(2), torch.randn(3, 5))
            ji_adf_loss(output, torch.tensor([0, 1, 2])).backward()
            self.assertTrue(torch.equal(before, model.clinical.backbone[1].running_mean))
            torch.testing.assert_close(output["log_probs"].exp().sum(1), torch.ones(3))
            for head in (model.image_head, model.metadata_head, model.joint_head):
                self.assertIsNotNone(head.weight.grad)
            self.assertIsNotNone(model.clinical.backbone[0].weight.grad)
            if variant in ("ji_adf", "concat_adaptive"):
                self.assertIsNotNone(model.decision_gate[0].weight.grad)
            else:
                torch.testing.assert_close(output["branch_weights"], torch.full((3, 3), 1/3))

    def test_resume_is_bitwise_equivalent_and_recipe_cannot_change(self):
        from ji_adf.training import train
        from utils.checkpointing import save_training
        reference, images = self.fixture()
        uninterrupted = self.args(reference, images, parent="uninterrupted")
        interrupted = self.args(reference, images, parent="interrupted")
        train(uninterrupted, model_factory=TinyJIADF)

        def stop(*args, **kwargs):
            save_training(*args, **kwargs)
            raise KeyboardInterrupt()

        with patch("ji_adf.training.save_training", side_effect=stop):
            with self.assertRaises(KeyboardInterrupt):
                train(interrupted, model_factory=TinyJIADF)
        train(interrupted, model_factory=TinyJIADF)
        left = torch.load(uninterrupted.output / "resume.pt", weights_only=False)
        right = torch.load(interrupted.output / "resume.pt", weights_only=False)
        for key in left["model"]:
            self.assertTrue(torch.equal(left["model"][key], right["model"][key]), key)
        self.assertEqual(left["scheduler"], right["scheduler"])
        interrupted.initialization = "v1"
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            train(interrupted, model_factory=TinyJIADF)

    def test_evaluation_freeze_and_upload_inference_agree(self):
        from improved.data import paired_frame, original_path
        from ji_adf.training import train
        from ji_adf.evaluation import evaluate
        from ji_adf.baseline import freeze_run
        from ji_adf.runtime import capture_provenance, write_json, output_directory
        from ji_adf.predictor import JIADFPredictor
        from utils.baselines import verify_baseline
        reference, images = self.fixture()
        variants = ["ji_adf", "concat_fixed"]
        for variant in variants:
            args = self.args(reference, images, variant)
            train(args, model_factory=TinyJIADF)
        run = self.root / "run"
        capture_provenance(run, reference)
        write_json(run / "run_plan.json", {"primary": "ji_adf", "variants": variants, "seeds": [42]})
        args = argparse.Namespace(reference=reference, images=images, output=run, device="cpu", batch_size=4,
                                  cache=None, reproduce_baseline=False, bootstrap=30)
        result = evaluate(args, model_factory=TinyJIADF)
        self.assertEqual(result["primary"], "ji_adf")
        self.assertIn("concat_fixed", result["ji_adf_minus_controls"])
        frozen = self.root / "saved_references/ji-adf-v1"
        freeze_run(run, frozen)

        def make_writable():
            frozen.chmod(0o755)
            for path in frozen.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)
        self.addCleanup(make_writable)
        self.assertTrue(verify_baseline(frozen)["intact"])
        self.assertFalse((frozen / "ji_adf/seed-42/resume.pt").exists())
        with self.assertRaisesRegex(ValueError, "frozen baseline"):
            output_directory(frozen / "new-run", reference)
        with self.assertRaises(ValueError):
            freeze_run(run, frozen)
        pair = paired_frame(reference / "split.csv").query("split == 'holdout'").iloc[0]
        with patch("ji_adf.predictor.JIADFModel", TinyJIADF), patch("ji_adf.predictor.InputValidator"):
            predictor = JIADFPredictor(frozen, device="cpu")
            result = predictor.analyze(original_path(images, pair.lesion_id, pair.clinical_id),
                                       original_path(images, pair.lesion_id, pair.dermoscopic_id), pair.to_dict())
        from components.paths import DIAGNOSIS_COLS
        expected = pd.read_csv(run / "predictions/ji_adf.holdout.csv").set_index("lesion_id").loc[pair.lesion_id]
        np.testing.assert_allclose(result["probabilities"], expected[["prob_" + name for name in DIAGNOSIS_COLS]].to_numpy(float),
                                   rtol=1e-5, atol=1e-6)
        # A fake metric must not be promoted into the new baseline.
        report_path = run / "metrics/ji_adf.holdout.json"
        report = json.loads(report_path.read_text())
        report["macro_f1"] = 0.99
        write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, "Recorded metrics differ"):
            freeze_run(run, self.root / "bad-reference")

    def test_rejection_precedes_preprocessing_and_disease_forward(self):
        from ji_adf.predictor import JIADFPredictor
        from utils.input_validation import InputRejected, ValidationResult
        predictor = JIADFPredictor.__new__(JIADFPredictor)
        predictor.validator = Mock()
        predictor.validator.require.side_effect = [None, InputRejected(ValidationResult("unsupported_image", "No lesion"), "bottle")]
        with patch("ji_adf.predictor.rgb_square") as preprocessing:
            with self.assertRaises(InputRejected):
                predictor.analyze("lesion", "bottle")
            preprocessing.assert_not_called()
        self.assertEqual(predictor.validator.require.call_count, 2)

    def test_cancel_stops_both_real_child_processes(self):
        import ji_adf.workflow as workflow
        reference, images = self.fixture()
        args = self.args(reference, images)
        args.output = self.root / "cancel-run"
        args.seeds, args.variants = [42], ["ji_adf", "concat_fixed"]
        original_popen, children = subprocess.Popen, []

        def child(_command, **kwargs):
            process = original_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            children.append(process)
            return process

        clock = Mock(monotonic=time.monotonic, sleep=Mock(side_effect=KeyboardInterrupt()))
        try:
            with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2), \
                    patch("torch.cuda.get_device_name", return_value="test GPU"), patch.object(workflow, "time", clock), \
                    patch.object(workflow.subprocess, "Popen", side_effect=child):
                with self.assertRaises(KeyboardInterrupt):
                    workflow.run(args)
            self.assertEqual(len(children), 2)
            self.assertTrue(all(process.poll() is not None for process in children))
        finally:
            for process in children:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_notebook_cells_compile_and_default_primary_is_explicit(self):
        notebook = json.loads((ROOT / "notebooks/kaggle_ji_adf.ipynb").read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{index}", "exec")
        self.assertIn("VARIANTS = ['ji_adf', 'concat_fixed']", "".join(notebook["cells"][1]["source"]))


if __name__ == "__main__":
    unittest.main()
