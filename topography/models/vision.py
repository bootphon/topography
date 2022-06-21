"""Provides ResNet18 model."""
from torch import nn
from torchvision import models


def resnet18(num_classes: int = 10, **kwargs):
    """Constructs a ResNet-18 for CIFAR-10 or CIFAR-100.
    Change the `conv1` layer: the kernel size is reduced from 7 to 3,
    the strides from 2 to 1, and the padding from
    3 to 1. It is better suited for CIFAR and follows [resnet]_ architecture.

    Parameters
    ----------
    num_classes : int
        Number of classes in the training set. Would be 10 or 100
        if trained on CIFAR.

    **kwargs :
        Optional arguments in torchvision.models.ResNet.

    Returns
    -------
    ResNet :
        Instance of ResNet-18

    References
    ----------
    .. [resnet] https://arxiv.org/abs/1512.03385
    """
    resnet = models.resnet18(
        pretrained=False, num_classes=num_classes, **kwargs
    )
    resnet.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    return resnet


def vgg16_bn(num_classes: int = 10, **kwargs):
    """Builds a VGG16 with batch normalization. Same function
    as in torchvision.models, but it does not use weights from
    pretrained models.

    Returns
    -------
    VGG
        VGG16 instance with batch normalization.
    """
    return models.vgg16_bn(pretrained=False, num_classes=num_classes, **kwargs)


def alexnet(num_classes: int = 10, dropout: float = 0.5):
    """Builds an AlexNet. Same function as in torchvision.models,
    but it does not use weights from pretrained models.

    Returns
    -------
    AlexNet
        AlexNet instance.
    """
    return models.AlexNet(num_classes=num_classes, dropout=dropout)
