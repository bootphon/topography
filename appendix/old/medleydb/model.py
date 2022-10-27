"""Provides a model for estimating fundamental frequencies.
"""

import torch
from torch import nn


class DeepSalienceCNN(nn.Module):
    """Model from [deepsalience]_.

    References
    ----------
    .. [deepsalience] http://www.justinsalamon.com/uploads/4/3/9/4/4394963/bittner_deepsalience_ismir_2017.pdf # pylint: disable=line-too-long # noqa: E501
    """

    def __init__(self) -> None:
        """Create the model. The parameters of the convolutional layers
        are already set and can not be changed.
        """
        super().__init__()
        num_layers = 6
        channels = [6, 128, 64, 64, 64, 8, 1]
        kernels = [(5, 5), (5, 5), (3, 3), (3, 3), (70, 3), (1, 1)]

        layers = []
        for i in range(num_layers):
            in_channels, out_channels = channels[i], channels[i + 1]
            batch_norm = nn.BatchNorm2d(out_channels)
            conv = nn.Conv2d(
                in_channels, out_channels, kernels[i], padding="same"
            )
            if i == num_layers - 1:
                layers.append(nn.Sequential(conv, batch_norm))
            else:
                layers.append(
                    nn.Sequential(conv, batch_norm, nn.ReLU(inplace=True))
                )
        self.layers = nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Apply the model.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.layers(inp).squeeze(1)
