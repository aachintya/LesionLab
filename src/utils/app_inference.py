"""Single-lesion inference for the local UI; validate originals before classification."""
from __future__ import annotations

from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd
import torch

from components.augmentations import get_transforms
from components.image_cache import load_preprocessed_image
from components.metadata_encoder import MetadataEncoder
from components.paths import DIAGNOSIS_COLS, NUM_CLASSES
from models.efficientnet_v2_image import build_image_model
from models.fusion_lr import load_fusion_lr
from models.metadata_mlp import build_metadata_model
from utils.input_validation import InputValidator
from utils.result_interpretation import interpret_result
from utils.screening import screening_from_result, screening_scores, presentation_screening
from utils.training import pick_device

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
MODE_FILES = {
    "clinical": ("best_clinical.pt",),
    "dermoscopic": ("best_dermoscopic.pt",),
    "fusion": ("best_clinical.pt", "best_dermoscopic.pt", "best_metadata.pt",
               "metadata_encoder.json", "best_fusion_lr.joblib"),
    "validator": (),
}
CLASS_LABELS = (
    "Actinic keratosis / intraepidermal carcinoma", "Basal cell carcinoma",
    "Other benign proliferations", "Benign keratinocytic lesion", "Dermatofibroma",
    "Inflammatory / infectious condition", "Other malignant proliferations", "Melanoma",
    "Melanocytic nevus", "Squamous cell carcinoma / keratoacanthoma",
    "Vascular lesion / hemorrhage",
)


def missing_checkpoints(directory=CHECKPOINT_DIR, mode="fusion"):
    if mode not in MODE_FILES:
        raise ValueError("Choose clinical, dermoscopic, fusion, or validator mode.")
    return [name for name in ("input_validator.pt", *MODE_FILES[mode])
            if not (Path(directory) / name).is_file()]


def checkpoint_signature(directory=CHECKPOINT_DIR):
    """Invalidate cached resources when an imported checkpoint changes."""
    files = sorted({"input_validator.pt", *MODE_FILES["fusion"]})
    return tuple((name, (Path(directory) / name).stat().st_mtime_ns,
                  (Path(directory) / name).stat().st_size)
                 for name in files if (Path(directory) / name).is_file())


class LesionPredictor:
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR, device="auto", *, ensemble_dir=None):
        self.directory = Path(checkpoint_dir)
        self.device = pick_device(device)
        # A rejected image never needs to allocate a disease model on the accelerator.
        self.validator = InputValidator(self.directory / "input_validator.pt", device="cpu")
        self.models = {}
        self.lock = RLock()
        self.ensemble_dir = Path(ensemble_dir) if ensemble_dir is not None else None
        self.ensemble = None

    def _image_model(self, modality):
        if modality not in self.models:
            model = build_image_model(NUM_CLASSES, dropout=0.5, pretrained=False)
            state = torch.load(self.directory / f"best_{modality}.pt",
                               map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            self.models[modality] = model.to(self.device).requires_grad_(False).eval()
        return self.models[modality]

    def _image_input(self, path, modality):
        image = load_preprocessed_image(path, modality=modality, roi_mode="foreground")
        from PIL import Image
        _, transform = get_transforms(modality=modality)
        inputs = transform(Image.fromarray(image)).unsqueeze(0).to(self.device)
        return image, inputs

    def _image_probs(self, path, modality):
        model = self._image_model(modality)
        _, inputs = self._image_input(path, modality)
        with torch.inference_mode():
            logits = model(inputs)
            return self._probabilities(torch.softmax(logits, dim=1).cpu().numpy()[0])

    def _image_evidence(self, path, modality):
        from utils.demo_ensemble import image_views
        model = self._image_model(modality)
        _, inputs = self._image_input(path, modality)
        with torch.inference_mode():
            features = model.backbone(inputs)
            original = torch.softmax(model.head(features), dim=1)
            total = original.clone()
            for view in image_views(inputs)[1:]:
                total += torch.softmax(model(view), dim=1)
            return (self._probabilities(original.cpu().numpy()[0]),
                    features.cpu().numpy()[0],
                    self._probabilities((total / 6).cpu().numpy()[0]))

    def _image_heatmap(self, path, modality, target_class):
        from utils.image_explanations import grad_cam
        image, inputs = self._image_input(path, modality)
        return grad_cam(self._image_model(modality), inputs, image, target_class=target_class)

    @staticmethod
    def _probabilities(values):
        values = np.asarray(values, dtype=np.float64)
        if (values.shape != (NUM_CLASSES,) or not np.isfinite(values).all()
                or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-4)):
            raise ValueError("The model returned invalid scores. Classification was withheld.")
        return values

    def _metadata_input(self, metadata):
        if "metadata" not in self.models:
            self.encoder = MetadataEncoder.load(self.directory / "metadata_encoder.json")
            model = build_metadata_model(self.encoder.input_dim_, NUM_CLASSES)
            model.load_state_dict(torch.load(self.directory / "best_metadata.pt",
                                            map_location="cpu", weights_only=True))
            self.models["metadata"] = model.to(self.device).requires_grad_(False).eval()
        # Never pass diagnosis, confirmation method, or post-biopsy features to the encoder.
        row = {name: metadata.get(name) for name in ("age_approx", "skin_tone_class", "sex", "site")}
        for name, upper in (("age_approx", 120), ("skin_tone_class", 5)):
            if row[name] is not None:
                value = float(row[name])
                if not np.isfinite(value) or not 0 <= value <= upper:
                    raise ValueError(f"Invalid {name}.")
                row[name] = value
        return self.encoder.transform(pd.DataFrame([row]))

    def _metadata_probs(self, metadata):
        features = self._metadata_input(metadata)
        with torch.inference_mode():
            logits = self.models["metadata"](torch.from_numpy(features).to(self.device))
            return self._probabilities(torch.softmax(logits, dim=1).cpu().numpy()[0])

    def _original_fusion(self, branches):
        if "fusion" not in self.models:
            fusion = load_fusion_lr(self.directory / "best_fusion_lr.joblib")
            if list(fusion.classes_) != list(range(NUM_CLASSES)) or fusion.n_features_in_ != 33:
                raise ValueError("Fusion checkpoint does not match this 11-class project.")
            self.models["fusion"] = fusion
        vector = np.concatenate([branches[name] for name in ("clinical", "dermoscopic", "metadata")])
        return self._probabilities(self.models["fusion"].predict_proba(vector[None, :])[0])

    def analyze(self, images, *, mode="clinical", metadata=None, include_heatmaps=False):
        if mode not in MODE_FILES:
            raise ValueError("Unsupported analysis mode.")
        required = ("clinical", "dermoscopic") if mode == "fusion" else (mode,)
        if mode == "validator":
            required = ("image",)
        if any(name not in images for name in required):
            raise ValueError("Provide both photographs of the same lesion for multimodal analysis."
                             if mode == "fusion" else "Upload an image first.")
        with self.lock:
            # All originals must pass before either image is preprocessed or classified.
            validations = {name: self.validator.require(images[name]).to_dict() for name in required}
            if mode == "validator":
                return {"mode": mode, "input_validation": validations, "scores": []}
            missing = missing_checkpoints(self.directory, mode)
            if missing:
                raise FileNotFoundError("Import your trained Kaggle models in Setup. Missing: " + ", ".join(missing))
            ensemble_details = {}
            if mode == "fusion" and self.ensemble_dir is not None:
                from utils.demo_ensemble import FrozenFeatureEnsemble, ENSEMBLE_ID, TTA_VIEWS
                if self.ensemble is None:
                    self.ensemble = FrozenFeatureEnsemble(self.directory, self.ensemble_dir)
                evidence = {name: self._image_evidence(images[name], name) for name in required}
                branches = {name: values[0] for name, values in evidence.items()}
                branches["metadata"] = self._metadata_probs(metadata or {})
                tta = {name: values[2] for name, values in evidence.items()}
                scores, members = self.ensemble.predict(
                    {name: values[1] for name, values in evidence.items()},
                    self._metadata_input(metadata or {})[0], branches, tta)
                scores = self._probabilities(scores)
                ensemble_details = {
                    "pipeline": {"id": ENSEMBLE_ID, "label": "Frozen-feature + six-view ensemble",
                                 "combination": "equal geometric mean", "tta_views": list(TTA_VIEWS),
                                 "calibrated": False, "metadata_features": 12, "feature_dimensions": 173},
                    "ensemble_members": {key: list(map(float, self._probabilities(value))) for key, value in members.items()},
                    "image_tta_scores": {key: list(map(float, value)) for key, value in tta.items()},
                    "branch_score_scope": "Single-view original specialists; ensemble components are reported separately."}
            elif mode == "fusion":
                branches = {name: self._image_probs(images[name], name) for name in required}
                branches["metadata"] = self._metadata_probs(metadata or {})
                scores = self._original_fusion(branches)
            else:
                branches = {name: self._image_probs(images[name], name) for name in required}
                scores = branches[mode]
            best = int(np.argmax(scores))
            result = {"mode": mode, "class_code": DIAGNOSIS_COLS[best], "class_name": CLASS_LABELS[best],
                    "device": str(self.device), "input_validation": validations,
                    "scores": [{"class": code, "name": label, "score": float(score)}
                               for code, label, score in zip(DIAGNOSIS_COLS, CLASS_LABELS, scores)],
                    "branch_scores": {name: list(map(float, values)) for name, values in branches.items()},
                    "pipeline": {"id": "original_score_fusion" if mode == "fusion" else f"original_{mode}",
                                 "label": "Saved original fusion" if mode == "fusion" else "Single-view specialist"},
                    **ensemble_details}
            result["interpretation"] = interpret_result(result, metadata=metadata)
            result["screening"] = screening_from_result(result)
            result["presentation_screening"] = presentation_screening(result)
            if mode == "fusion" and self.ensemble_dir is not None:
                reference = self._original_fusion(branches)
                target = int(np.argmax(reference))
                result["baseline_comparison"] = {"pipeline": "original_score_fusion",
                    "class_code": DIAGNOSIS_COLS[target], "class_name": CLASS_LABELS[target],
                    "scores": list(map(float, reference)), "screening": screening_scores(reference)}
            result["heatmaps_requested"] = bool(include_heatmaps)
            result["heatmaps"] = {}
            if include_heatmaps:
                for name in required:
                    target = int(np.argmax(branches[name]))
                    try:
                        explanation = self._image_heatmap(images[name], name, target)
                    except Exception as exc:
                        # An explanation failure must not turn into an invented map
                        # or discard an otherwise valid, already checked prediction.
                        explanation = {"status": "unavailable", "message": "The visual explanation could not be generated for this view.",
                                       "error_type": type(exc).__name__}
                    result["heatmaps"][name] = {**explanation, "class_code": DIAGNOSIS_COLS[target],
                                               "class_name": CLASS_LABELS[target]}
            return result
