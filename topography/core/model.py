"""Topographic model. Add positions to all Conv2d layers in a base model
and try to enforce topographic organization.
The inverse pairwise distances between channels are stored under
the `inverse_distance` buffer.
"""
import copy
from collections import OrderedDict
from operator import attrgetter
from typing import Callable, List, Optional

import torch
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
        position_scheme: str = "hypercube",
        topographic_layer_names: Optional[List[str]] = None,
    ) -> None:
        """Creates the model. It is the same as the original model,
        but adds an `activations` field that records outputs
        of each Conv2d layer. Also adds an `inverse_distance`
        buffer to them.

        Parameters
        ----------
        model : nn.Module
            Original model. Must have Conv2d layers.
        dimension : int, optional
            Dimension of the position assigned to each channel
            of each Conv2d layer, by default 2.
        norm : str, optional
            Which norm between positions to use. Must be "euclidean" or "l1",
            by default "euclidean".
        position_scheme : str, optional
            How to assign positions. Must be "hypercube", by default
            "hypercube".
        topographic_layer_names : List[str], optional
            List of names of the Conv2d layers to which add topography.
            If None, all Conv2d layers will be used. By default None.
        """
        super().__init__()
        self.model = copy.deepcopy(model)

        def condition(name: str, layer: nn.Module) -> bool:
            is_conv2d = isinstance(layer, nn.Conv2d)
            if topographic_layer_names is None:
                return is_conv2d
            return is_conv2d and name in topographic_layer_names

        names, conv_layers = zip(
            *[
                (name, layer)
                for name, layer in self.model.named_modules()
                if condition(name, layer)
            ]
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
                self.activations[name] = out

            return hook

        self.topographic_layer_names = names
        for name, layer in zip(names, conv_layers):
            layer.register_buffer("inverse_distance", inv_dist[name])
            layer.register_forward_hook(get_activation(name))

    @property
    def inverse_distance(self) -> OrderedDict:
        """Get the inverse distances between the channels positions
        of each Conv2d layer.
        For each layer, a buffer "inverse_distance" is registered, and we
        retrieve them in this method.
        It is implemented that way in order for them to be inside
        the `state_dict` and to move to the same device as the model when
        needed.

        Returns
        -------
        OrderedDict
            The dictionnary of inverse distances.
        """
        inv_dist = OrderedDict()
        for name in self.topographic_layer_names:
            inv_dist[name] = attrgetter(name)(self.model).inverse_distance
        return inv_dist

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
