"""Provides different models.
"""
import torch.nn as nn
from torchvision import models


def resnet(out_features: int = 1000, pretrained: bool = True,
           num_layers: int = 18) -> models.ResNet:
    """ResNet model

    Parameters
    ----------
    out_features : int, optional
        Number of classes, by default 1000.
    pretrained : bool, optional
        Whether to load a pretrained model from torchvision, by default True.
    num_layers : int, optional
        Number of layers, by default 18.

    Returns
    -------
    models.ResNet
        Desired model.

    Raises
    ------
    ValueError
        If num_layers is an unknown size, i.e. it is not in
        (18, 34, 50, 101, 152).
    """
    if num_layers not in [18, 34, 50, 101, 152]:
        raise ValueError('Invalid resnet model.')
    model = getattr(models, f'resnet{num_layers}')(pretrained=pretrained)
    if out_features != model.fc.out_features:
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=out_features)
    return model
