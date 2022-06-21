from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from topography.core.distance import _POSITIONS
from topography.core.loss import _channel_correlation

FigureAxes = Tuple[Figure, Axes]


def plot_activations(
    activations: torch.Tensor,
    norm: bool = True,
    position_scheme: str = "cube",
    **kwargs
) -> FigureAxes:
    normalize = (
        Normalize(vmin=activations.min(), vmax=activations.max())
        if norm
        else None
    )
    positions = _POSITIONS[position_scheme](
        num=activations.shape[0], dimension=2, integer_positions=True
    )
    num_axis = positions.max() + 1
    fig, ax = plt.subplots(
        nrows=num_axis, ncols=num_axis, figsize=(2 * num_axis, 2 * num_axis)
    )
    for k, (i, j) in enumerate(positions):
        im = ax[i, j].imshow(activations[k], norm=normalize, **kwargs)
        ax[i, j].axis("off")
    if norm:
        fig.colorbar(im, ax=ax.ravel().tolist())
    return fig, ax


def plot_correlation_matrix(
    batch_activations: torch.Tensor,
    idx: int,
    figsize: Tuple[int, int] = (10, 10),
    **kwargs
) -> FigureAxes:
    correlation = _channel_correlation(batch_activations, eps=1e-8)
    sym_correlation = (correlation[idx] + correlation[idx].T).fill_diagonal_(1)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sym_correlation, vmin=-1, vmax=1, **kwargs)
    fig.colorbar(im, ax=ax)
    return fig, ax


def plot_aggregated_correlations(
    agg_correlations: Dict[float, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    **kwargs
) -> FigureAxes:
    distance = sorted(list(agg_correlations.keys()))
    mean_corr = [np.mean(agg_correlations[dist]) for dist in distance]
    std_corr = [np.std(agg_correlations[dist]) / 2 for dist in distance]
    ref = np.linspace(min(distance), max(distance))
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(distance, mean_corr, std_corr, **kwargs)
    ax.plot(ref, 1 / (ref + 1), label=r"$\frac{1}{x+1}$")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Correlation")
    return fig, ax


def aggregate_correlation(model, loader, layer, device):
    agg = {}
    inv_dist = model.inverse_distance[layer].cpu()
    distance = torch.round(
        1 / ((inv_dist + inv_dist.T).fill_diagonal_(1)) - 1, decimals=3
    ).numpy()

    for batch, _ in loader:
        model(batch.to(device))
        activ = model.activations[layer]
        correlations = _channel_correlation(activ, 1e-8)
        for b in range(correlations.shape[0]):
            correlations[b].fill_diagonal_(1)
            for i in range(correlations.shape[1]):
                for j in range(i, correlations.shape[1]):
                    if distance[i, j] not in agg:
                        agg[distance[i, j]] = []
                    agg[distance[i, j]].append(correlations[b, i, j].item())
    return agg
