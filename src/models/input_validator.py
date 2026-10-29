"""Binary supported-lesion photograph classifier (1 = supported, 0 = unrelated)."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_input_validator(*, pretrained: bool = False):
    # Inference must never download weights or silently use random initialization.
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def validator_transform():
    return ResNet18_Weights.DEFAULT.transforms()
