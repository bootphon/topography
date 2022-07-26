"""Provides models to use on audio and speech datasets."""
from torch import nn
from torchvision import models


def speech_vgg(num_classes: int = 35, base: str = "vgg16_bn") -> models.VGG:
    """Speech VGG. The only difference with VGG models
    from torchvision is that the first layer expects
    the input tensor to have one channel and not three.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the dataset, by default 35.
    base : str, optional
        Base VGG model to wrap. Must be accessed in
        torchvision.models, by default "vgg16_bn".

    Returns
    -------
    VGG
        Speech VGG.
    """
    model = getattr(models, base)(num_classes=num_classes, pretrained=False)
    model.features[0] = nn.Conv2d(
        in_channels=1,
        out_channels=model.features[0].out_channels,
        kernel_size=model.features[0].kernel_size,
        stride=model.features[0].stride,
        padding=model.features[0].padding,
    )
    return model
