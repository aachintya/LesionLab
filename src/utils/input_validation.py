"""Fail-closed input validation on original images, before lesion preprocessing."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math

from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_VALIDATOR = Path(__file__).resolve().parents[1] / "checkpoints/input_validator.pt"


@dataclass(frozen=True)
class ValidationResult:
    status: str
    message: str
    supported_probability: float | None = None

    @property
    def accepted(self):
        return self.status == "accepted"

    def to_dict(self):
        return asdict(self)


class InputRejected(ValueError):
    def __init__(self, result, path):
        self.result = result
        super().__init__(f"{path}: {result.status}: {result.message}")


class InputValidator:
    def __init__(self, checkpoint=DEFAULT_VALIDATOR, *, device="cpu"):
        import torch
        from models.input_validator import build_input_validator, validator_transform

        self.device = torch.device(device)
        self.model = None
        self.transform = validator_transform()
        path = Path(checkpoint)
        if not path.is_file():
            return
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported input-validator checkpoint format")
        self.threshold = float(payload["threshold"])
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("Invalid input-validator threshold")
        self.model = build_input_validator().to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def validate(self, path):
        try:
            with Image.open(path) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            return ValidationResult("invalid_image", "Upload a readable image file.")
        if min(image.size) < 64:
            return ValidationResult("poor_quality", "Upload an image at least 64 pixels on each side.")
        if self.model is None:
            return ValidationResult("validator_unavailable", "A trained input validator is required before prediction.")
        import torch
        with torch.inference_mode():
            score = float(torch.sigmoid(self.model(self.transform(image).unsqueeze(0).to(self.device))).item())
        if not math.isfinite(score):
            return ValidationResult("validator_unavailable", "Input validation failed; prediction withheld.")
        if score < self.threshold:
            return ValidationResult("unsupported_image", "Upload a clear clinical or dermoscopic photograph of a skin lesion.", score)
        return ValidationResult("accepted", "Supported lesion photograph.", score)

    def require(self, path):
        result = self.validate(path)
        if not result.accepted:
            raise InputRejected(result, path)
        return result
