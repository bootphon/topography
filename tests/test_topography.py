"""Test of the TopographicModel and the TopographicLoss."""
import pytest
import torch
from torch import nn

import topography
from topography import models, topographic_loss
from topography.core.loss import _reduce


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_topographic_loss(reduction):
    model = topography.TopographicModel(models.resnet18())
    loss = topography.TopographicLoss(reduction=reduction)
    sample = torch.rand(1, 3, 32, 32)
    model(sample)
    loss(model.activations, model.inverse_distance)


def test_topographic_loss_bad_reduction():
    with pytest.raises(ValueError) as err:
        topography.TopographicLoss(reduction="bad")
    assert "is not a valid value for reduction" in str(err.value)

    activations = {"layer": torch.rand(3, 3)}
    inverse_distances = {"layer": torch.rand(3, 3)}
    with pytest.raises(ValueError) as err:
        topographic_loss(activations, inverse_distances, reduction="bad")
    assert "is not a valid value for reduction" in str(err.value)

    with pytest.raises(ValueError) as err:
        _reduce(torch.zeros(2, 2), reduction="bad")
    assert "is not a valid value for reduction" in str(err.value)


def test_wrong_model_name():
    with pytest.raises(ValueError) as err:
        models.topographic_layer_names("efficientnet")
    assert str(err.value).startswith("Unknown model")


@pytest.mark.parametrize("model_name", ["resnet18", "vgg16_bn", "densenet121"])
def test_layer_names(model_name):
    names = models.topographic_layer_names(model_name)
    model = topography.TopographicModel(
        getattr(models, model_name)(), topographic_layer_names=names
    )
    if names is None:
        names = tuple(
            name
            for name, module in model.model.named_modules()
            if isinstance(module, nn.Conv2d)
        )
    assert model.topographic_layer_names == names
    assert tuple(model.inverse_distance.keys()) == names
