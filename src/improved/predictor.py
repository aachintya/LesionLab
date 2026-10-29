"""Paired v2 upload inference with the original fail-closed lesion validator."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from components.metadata_encoder import MetadataEncoder
from components.paths import DIAGNOSIS_COLS
from improved.data import encode_metadata, image_transform, rgb_square
from improved.evaluation import ensemble_probabilities
from improved.model import PairedLesionModel
from utils.input_validation import InputValidator
from utils.training import _autocast, pick_device


class PairedPredictor:
    def __init__(self, run_dir, *, device="auto"):
        self.directory = Path(run_dir)
        self.selection = json.loads((self.directory / "selection.json").read_text())
        self.device = pick_device(device)
        from utils.baselines import sha256_file
        reference_manifest = json.loads((self.directory / "reference_manifest.json").read_text())
        gate = self.directory / "input_validator.pt"
        if (not gate.is_file() or sha256_file(gate) != reference_manifest["sha256"]["checkpoints/input_validator.pt"]):
            raise ValueError("The original input validator is missing or changed")
        self.validator = InputValidator(self.directory / "input_validator.pt", device="cpu")
        self.models = []
        for name, expected in sorted(self.selection["signature"]["checkpoints"].items()):
            run = self.directory / name
            if sha256_file(run / "best.pt") != expected:
                raise ValueError("A v2 checkpoint changed after ensemble selection")
            saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            if sha256_file(run / "metadata_encoder.json") != saved["config"]["metadata_encoder_sha256"]:
                raise ValueError("The metadata encoder changed after training")
            model = PairedLesionModel(saved["config"]["metadata_dim"]).to(self.device).eval()
            model.load_state_dict(saved["model"], strict=True)
            self.models.append((model, MetadataEncoder.load(run / "metadata_encoder.json"), saved["config"]["size"]))
        self.counts = np.asarray(self.selection["train_class_counts"])

    def analyze(self, clinical_path, dermoscopic_path, metadata=None):
        # Validate BOTH originals before preprocessing or any disease-model forward.
        self.validator.require(clinical_path)
        self.validator.require(dermoscopic_path)
        frame = pd.DataFrame([metadata or {}]).reindex(columns=["age_approx", "skin_tone_class", "sex", "site"])
        all_logits = []
        with torch.inference_mode():
            for model, encoder, size in self.models:
                transform = image_transform(size, False)
                images = [transform(rgb_square(path, max(384, size))).unsqueeze(0).to(self.device)
                          for path in (clinical_path, dermoscopic_path)]
                encoded = torch.from_numpy(encode_metadata(encoder, frame)).to(self.device)
                with _autocast(self.device, self.device.type == "cuda"):
                    output = model(*images, encoded)["logits"]
                all_logits.append(output.float().cpu().numpy())
        probabilities = ensemble_probabilities(np.stack(all_logits), self.counts, self.selection["alpha"])[0]
        return {"class_order": list(DIAGNOSIS_COLS), "predicted": DIAGNOSIS_COLS[int(probabilities.argmax())],
                "probabilities": probabilities.tolist(), "input_gate_enabled": True,
                "warning": "Uncalibrated research output; not a medical diagnosis."}
