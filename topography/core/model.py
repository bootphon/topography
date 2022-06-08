"""Topographic model. Add positions to all Conv2d layers in a base model
and try to enforce topographic organization.
The inverse pairwise distances between channels are stored under
the `inverse_distance` buffer.
"""
import copy
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from topography.core.distance import inverse_distance


class TopographicModel(nn.Module):
    """Class to introduce topographic organization in PyTorch models
    with Conv2d layers."""

    def __init__(
        self,
        model: nn.Module,
        dimension: int = 2,
        norm: str = "euclidean",
        position_scheme: str = "cube",
    ) -> None:
        """Creates the model. It is the same as the original model,
        but adds an `activations` field that records outputs
        of each Conv2d layer. Also adds an `inverse_distance`
        buffer to them, and a `inverse_distance` field to the
        TopographicModel in order to get them directly.

        Parameters
        ----------
        model : nn.Module
            Original model. Must have Conv2d layers.
        dimension : int, optional
            Dimension of the position assigned to each channel
            of each Conv2d layer, by default 2.
        norm : str, optional
            Which norm between positions to use. Must be "euclidean",
            by default "euclidean".
        position_scheme : str, optional
            How to assign positions. Must be "cube", by default "cube".
        """
        super().__init__()
        self.model = copy.deepcopy(model)

        names, conv_layers = zip(
            *list(
                filter(
                    lambda named_module: isinstance(named_module[1], nn.Conv2d),
                    self.model.named_modules(),
                )
            )
        )
        out_channels = OrderedDict(
            zip(names, [layer.out_channels for layer in conv_layers])
        )

        self.activations = OrderedDict()
        inv_dist = inverse_distance(
            out_channels, dimension, norm, position_scheme
        )

        def get_activation(name: str) -> Callable[..., None]:
            def hook(_, __, out: torch.Tensor) -> None:
                self.activations[name] = F.relu(out)

            return hook

        self.inverse_distance = OrderedDict()
        for name, layer in zip(names, conv_layers):
            layer.register_buffer("inverse_distance", inv_dist[name])
            layer.register_forward_hook(get_activation(name))
            self.inverse_distance[name] = layer.inverse_distance

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed by the TopographicModel:
        it is the one done by the base model.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output computed by the base model.
        """
        return self.model(inp)
