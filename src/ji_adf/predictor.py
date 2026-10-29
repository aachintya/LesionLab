"""Use a completed JI-ADF run or frozen baseline with mandatory upload validation."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from components.metadata_encoder import MetadataEncoder
from components.paths import DIAGNOSIS_COLS
from improved.data import encode_metadata, image_transform, rgb_square
from ji_adf.model import JIADFModel, BRANCH_ORDER
from utils.baselines import sha256_file
from utils.input_validation import InputValidator
from utils.training import _autocast, pick_device


class JIADFPredictor:
    def __init__(self, directory, *, device="auto"):
        self.directory = Path(directory)
        schema = json.loads((self.directory / "inference.json").read_text())
        if schema["format_version"] != 3 or schema["variant"] != "ji_adf" or schema["class_order"] != list(DIAGNOSIS_COLS):
            raise ValueError("Incompatible JI-ADF inference schema")
        reference = json.loads((self.directory / "reference_manifest.json").read_text())
        gate = self.directory / "input_validator.pt"
        if not gate.is_file() or sha256_file(gate) != reference["sha256"]["checkpoints/input_validator.pt"]:
            raise ValueError("The original trained input validator is missing or changed")
        self.validator = InputValidator(gate, device="cpu")
        self.device = pick_device(device)
        self.members = []
        for member in schema["members"]:
            checkpoint, encoder_path = self.directory / member["checkpoint"], self.directory / member["encoder"]
            if not checkpoint.resolve().is_relative_to(self.directory.resolve()) or not encoder_path.resolve().is_relative_to(self.directory.resolve()):
                raise ValueError("Unsafe inference member")
            if sha256_file(checkpoint) != member["checkpoint_sha256"] or sha256_file(encoder_path) != member["encoder_sha256"]:
                raise ValueError("Inference checkpoint or metadata encoder changed")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            config = saved["config"]
            if (saved.get("format_version") != 3 or config["variant"] != "ji_adf"
                    or config["class_order"] != list(DIAGNOSIS_COLS)
                    or config["metadata_encoder_sha256"] != sha256_file(encoder_path)):
                raise ValueError("Checkpoint schema does not match the declared JI-ADF baseline")
            model = JIADFModel(config["metadata_dim"], variant="ji_adf").to(self.device).eval()
            model.load_state_dict(saved["model"], strict=True)
            self.members.append((model, MetadataEncoder.load(encoder_path), config["size"]))
        if not self.members:
            raise ValueError("An inference ensemble cannot be empty")

    def analyze(self, clinical_path, dermoscopic_path, metadata=None):
        self.validator.require(clinical_path)
        self.validator.require(dermoscopic_path)
        frame = pd.DataFrame([metadata or {}]).reindex(columns=["age_approx", "skin_tone_class", "sex", "site"])
        probabilities, weights = [], []
        with torch.inference_mode():
            for model, encoder, size in self.members:
                transform = image_transform(size, False)
                images = [transform(rgb_square(path, max(384, size))).unsqueeze(0).to(self.device)
                          for path in (clinical_path, dermoscopic_path)]
                encoded = torch.from_numpy(encode_metadata(encoder, frame)).to(self.device)
                with _autocast(self.device, False):
                    output = model(*images, encoded)
                if not all(torch.isfinite(value).all() for value in output.values()):
                    raise RuntimeError("Nonfinite JI-ADF prediction; no disease result was returned")
                probabilities.append(output["log_probs"].float().exp().cpu().numpy()[0])
                weights.append(output["branch_weights"].float().cpu().numpy()[0])
        prob = np.mean(probabilities, axis=0)
        return {"class_order": list(DIAGNOSIS_COLS), "probabilities": prob.tolist(),
                "predicted": DIAGNOSIS_COLS[int(prob.argmax())],
                "branch_weights": dict(zip(BRANCH_ORDER, np.mean(weights, axis=0).tolist())), "input_gate_enabled": True,
                "warning": "Uncalibrated research output; not a medical diagnosis. Branch weights are not causal explanations."}
