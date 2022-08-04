"""Test of the TopographicModel and the TopographicLoss."""
import pytest
import torch

from topography import TopographicLoss, TopographicModel, topographic_loss
from topography.core.loss import _reduce
from topography.models import resnet18, topographic_layer_names


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_topographic_loss(reduction):
    model = TopographicModel(resnet18())
    loss = TopographicLoss(reduction=reduction)
    sample = torch.rand(1, 3, 32, 32)
    model(sample)
    loss(model.activations, model.inverse_distance)


def test_topographic_loss_bad_reduction():
    with pytest.raises(ValueError) as err:
        TopographicLoss(reduction="bad")
    assert "is not a valid value for reduction" in str(err.value)

    activations = {"layer": torch.rand(3, 3)}
    inverse_distances = {"layer": torch.rand(3, 3)}
    with pytest.raises(ValueError) as err:
        topographic_loss(activations, inverse_distances, reduction="bad")
    assert "is not a valid value for reduction" in str(err.value)

    with pytest.raises(ValueError) as err:
        _reduce(torch.zeros(2, 2), reduction="bad")
    assert "is not a valid value for reduction" in str(err.value)


def test_wrong_topgraphic_model_name():
    with pytest.raises(ValueError) as err:
        topographic_layer_names("efficientnet")
    assert str(err.value).startswith("Unknown model")
