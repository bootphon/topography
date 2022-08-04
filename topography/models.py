"""Provides models to be trained of image (CIFAR-10, CIFAR-100) and
audio (BirdDCASE, SpeechCommands) datasets.
"""
from typing import List, Optional

from torch import nn
from torchvision import models


def resnet18(
    num_classes: int = 10, in_channels: int = 3, **kwargs
) -> models.ResNet:
    """ResNet-18 model.
    Changes the `conv1` layer: the kernel size is reduced from 7 to 3,
    the strides from 2 to 1, and the padding from
    3 to 1. It is better suited for CIFAR and follows [resnet]_ architecture.
    Can be set to be used with audio features by changing `in_channels`.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the training set. By default 10
    in_channels : int, optional
        Number of input channels (3 if RGB images, 1 if spectrogram, etc.).
        By default 3.
    **kwargs :
        Optional arguments in torchvision.models.resnet18.

    Returns
    -------
    ResNet :
        Instance of ResNet-18.

    References
    ----------
    .. [resnet] https://arxiv.org/abs/1512.03385
    """
    resnet = models.resnet18(num_classes=num_classes, **kwargs)

    resnet.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    return resnet


def vgg16_bn(
    num_classes: int = 10, in_channels: int = 3, **kwargs
) -> models.VGG:
    """VGG 16-layer model with batch normalization.
    Can be set to be used with audio features by changing `in_channels`.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the training set. By default 10
    in_channels : int, optional
        Number of input channels (3 if RGB images, 1 if spectrogram, etc.).
        By default 3.
    **kwargs :
        Optional arguments in torchvision.models.vgg16_bn.

    Returns
    -------
    VGG :
        VGG16 instance with batch normalization.
    """
    vgg = models.vgg16_bn(num_classes=num_classes, **kwargs)

    if in_channels != 3:
        vgg.features[0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=vgg.features[0].out_channels,
            kernel_size=vgg.features[0].kernel_size,
            stride=vgg.features[0].stride,
            padding=vgg.features[0].padding,
        )
        nn.init.kaiming_normal_(
            vgg.features[0].weight, mode="fan_out", nonlinearity="relu"
        )
        if vgg.features[0].bias is not None:
            nn.init.constant_(vgg.features[0].bias, 0)
    return vgg


def densenet121(
    num_classes: int = 10, in_channels: int = 3, **kwargs
) -> models.DenseNet:
    """Densenet-121 model.
    The first convolutional layer is modified: the kernel size is reduced
    from 7 to 3, the stride from 2 to 1 and the padding from 3 to 1.
    Can be set to be used with audio features by changing `in_channels`.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the training set. By default 10
    in_channels : int, optional
        Number of input channels (3 if RGB images, 1 if spectrogram, etc.).
        By default 3.
    **kwargs :
        Optional arguments in torchvision.models.densenet121.

    Returns
    -------
    models.DenseNet
        DenseNet-121 instance.
    """
    densenet = models.densenet121(num_classes=num_classes, **kwargs)

    densenet.features[0] = nn.Conv2d(
        in_channels,
        densenet.features[0].out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    nn.init.kaiming_normal_(densenet.features[0].weight)
    return densenet


def topographic_layer_names(model_name: str) -> Optional[List[str]]:
    """Returns the names of the Conv2d to which add topography.
    If None, all the Conv2d layers will be considered when using
    `TopographicModel`.

    Parameters
    ----------
    model_name : str
        Model name, must be either 'resnet18', 'densenet121' or 'vgg16_bn'.

    Returns
    -------
    Optional[List[str]]
        Module names in the model.

    Raises
    ------
    ValueError
        If the given name is invalid.
    """
    if model_name == "resnet18":
        return [
            "conv1",
            "layer1.1.conv2",
            "layer2.1.conv2",
            "layer3.1.conv2",
            "layer4.1.conv2",
        ]
    if model_name == "densenet121":
        return [
            "features.conv0",
            "features.denseblock1.denselayer6.conv2",
            "features.denseblock2.denselayer12.conv2",
            "features.denseblock3.denselayer24.conv2",
            "features.denseblock4.denselayer16.conv2",
        ]
    if model_name == "vgg16_bn":
        return None
    raise ValueError(
        f"Unknown model '{model_name}'. Model name must be"
        f" either 'resnet18', 'densenet121' or 'vgg16_bn'."
    )
