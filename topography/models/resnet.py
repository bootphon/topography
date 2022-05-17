"""Provides ResNet18 model."""
from torch import nn
from torchvision import models


def resnet18(**kwargs):
    """Constructs a ResNet-18 for CIFAR-10.
    Change the `conv1` layer: the kernel size is reduced from 7 to 3,
    the strides from 2 to 1, and the padding from
    3 to 1. It is better suited for CIFAR and follows [resnet]_ architecture.


    References
    ----------
    .. [resnet] https://arxiv.org/abs/1512.03385
    """
    resnet = models.resnet18(pretrained=False, **kwargs)
    resnet.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    return resnet
