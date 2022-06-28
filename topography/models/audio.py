"""Provides SpeechVGG model."""
from torch import nn
from torchvision import models


def speech_vgg(num_classes: int = 35, base: str = "vgg16_bn"):
    model = getattr(models, base)(num_classes=num_classes, pretrained=False)
    model.features[0] = nn.Conv2d(
        in_channels=1,
        out_channels=model.features[0].out_channels,
        kernel_size=model.features[0].kernel_size,
        stride=model.features[0].stride,
        padding=model.features[0].padding,
    )
    return model
