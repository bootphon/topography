""""Provides the topographic loss, following PyTorch API conventions."""
from collections import OrderedDict

import torch
from torch.nn.modules.loss import _Loss

from topography.base import MetricOutput, TensorDict


def _reduce(inp: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduce the input Tensor with the `reduction` method. See
    TopographicLoss for details on `reduction`.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor.
    reduction : str
        Reduction method.

    Returns
    -------
    torch.Tensor
        Reduced tensor.

    Raises
    ------
    ValueError
        If `reduction` is not a valid value for reduction.
    """
    if reduction == "none":
        output = inp
    elif reduction == "sum":
        output = inp.sum()
    elif reduction == "mean":
        output = inp.mean()
    else:
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return output


def _channel_correlation(activation: torch.Tensor, eps: float) -> torch.Tensor:
    """Computes correlation between channels of the output of a given layer.
    The correlation is defined as the cosine similarity between different
    channels, averaged over a given batch.

    Parameters
    ----------
    activation : torch.Tensor
        Output of the given layer, of shape (N, C, H, W).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Computed correlation of shape (N, C, C). The returned tensor
        is a batch of upper triangular matrices without the diagonal,
        in order to keep only the correlation between pairs of
        different channels.
    """
    flat_activ = activation.view(activation.shape[0], activation.shape[1], -1)
    channel_norm = torch.norm(flat_activ, dim=-1)
    normalization = torch.clamp(
        channel_norm[:, :, None] @ channel_norm[:, None, :], min=eps
    )
    full_corr = (flat_activ @ flat_activ.permute(0, 2, 1)) / normalization
    return torch.triu(full_corr, diagonal=1)


def topographic_loss(
    activations: TensorDict,
    inverse_distances: TensorDict,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> MetricOutput:
    """Functional implementation of the topographic loss.

    Parameters
    ----------
    activations : TensorDict
        Activations of each Conv2d layer.
    inverse_distances : TensorDict
        Inverse distances between positions of the channels
        for each Conv2d layer.
    eps : float, optional
        Small value to avoid division by zero when computing the
        correlation between channels, by default 1e-8.
    reduction : str, optional
        Specifies the reduction to apply to the output:
        `none`, `mean`,`sum` or `debug`. `none`: no reduction will
        be applied, `mean`: the mean of the output accross the batch
        is taken, `sum`: the output will be summed accross the batch,
        `debug`: a dictionnary with the loss for each layer is returnd.
        By default "mean".

    Returns
    -------
    MetricOutput
        Computed topographic loss.

    Raises
    ------
    ValueError
        If the reduction method is not `none`, `sum` or `mean`.
    """
    if reduction not in ["none", "mean", "sum", "debug"]:
        raise ValueError(
            f"Reduction method '{reduction}' is not available."
            "Must be either 'none', 'sum', 'mean' or 'debug'."
        )
    first_key = next(iter(activations))
    batch_size = activations[first_key].shape[0]
    loss = OrderedDict()
    total_loss = torch.zeros(batch_size).to(activations[first_key])
    for name, activation in activations.items():
        correlation = _channel_correlation(activation, eps)
        inv_dist = inverse_distances[name]
        layer_loss = ((correlation - inv_dist) ** 2).sum((1, 2))
        layer_loss /= correlation.shape[1] * (correlation.shape[1] - 1) // 2
        total_loss += layer_loss
        loss[f"topo-loss/{name}"] = _reduce(layer_loss, reduction).item()
    return MetricOutput(value=_reduce(total_loss, reduction), extras=loss)


class TopographicLoss(_Loss):
    """Creates a criterion that measures the topographic loss of
    a topographic model.
    """

    __constants__ = ["eps", "reduction"]
    eps: float

    def __init__(
        self,
        *,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        """Instantiates the topographic loss.

        Parameters
        ----------
        eps : float, optional
            Small value to avoid division by zero when computing the
            correlation between channels, by default 1e-8.
        reduction : str, optional
            Specifies the reduction to apply to the output:
            `none`, `mean`,`sum` or `debug`. `none`: no reduction will
            be applied, `mean`: the mean of the output is taken,
            `sum`: the output will be summed, `debug`: a dictionnary with
            the loss for each layer is returnd. By default "mean".

        Raises
        ------
        ValueError
            If the reduction method is not `none`, `sum`, `mean` or
           `debug`.
        """
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        super().__init__(None, None, reduction)
        self.eps = eps

    def forward(
        self, activations: TensorDict, inverse_distances: TensorDict
    ) -> MetricOutput:
        """Computes the topographic loss for a given topographic model.

        Parameters
        ----------
        activations : TensorDict
            Activations of each Conv2d layer.
        inverse_distances : TensorDict
            Inverse distances between positions of the channels
            for each Conv2d layer.

        Returns
        -------
        MetricOutput
            Computed topographic loss.
        """
        return topographic_loss(
            activations,
            inverse_distances,
            eps=self.eps,
            reduction=self.reduction,
        )
