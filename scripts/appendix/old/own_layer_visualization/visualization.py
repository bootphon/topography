"""Visualizing higher-layer features of a deep network
D Erhan, Y Bengio, A Courville, P Vincent
"""
import functools
from operator import attrgetter
from typing import Tuple

import PIL
import torch
from torch import nn
from torchvision import transforms

Normalization = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

_CIFAR_NORMALIZATION: Normalization = (
    (0.4914, 0.4822, 0.4465),
    (0.2023, 0.1994, 0.2010),
)


class VisualizationLayerCNN:
    def __init__(
        self,
        model: nn.Module,
        normalization: Normalization = _CIFAR_NORMALIZATION,
    ) -> None:
        self.model = model.eval()
        self._normalization = normalization
        self.transform = transforms.Normalize(*normalization)
        self.inverse_transform = transforms.Compose(
            [
                transforms.Normalize(
                    [0, 0, 0], [1 / std for std in normalization[1]]
                ),
                transforms.Normalize(
                    [-mean for mean in normalization[0]], [1, 1, 1]
                ),
                functools.partial(torch.clamp, min=0, max=1),
                functools.partial(torch.squeeze, dim=0),
                transforms.ToPILImage(),
            ]
        )
        self.output = None

    def _set_hook(self, layer_name: str, channel: int) -> None:
        def hook(module, grad_in, grad_out):
            self.output = grad_out[0, channel]

        attrgetter(layer_name)(self.model).register_forward_hook(hook)

    def __call__(
        self,
        layer_name: str,
        channel: int,
        n_iter: int = 30,
        shape: Tuple = (3, 32, 32),
        device: str = "cpu",
    ) -> PIL.Image.Image:
        self._set_hook(layer_name, channel)
        inp = torch.rand(1, *shape, device=device)
        img = self.transform(inp).clone().detach().requires_grad_()
        optimizer = torch.optim.Adam([img], lr=0.1)
        for _ in range(n_iter):
            self.model(img)
            loss = -torch.mean(self.output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self.inverse_transform(img)
