from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from tensorboard.backend.event_processing import event_accumulator

from topography.core.distance import _POSITIONS
from topography.core.loss import _channel_correlation

FigureAxes = Tuple[Figure, Axes]


def tensorboard_to_dataframe(experiment_id: str) -> pd.DataFrame:
    event_acc = event_accumulator.EventAccumulator(experiment_id)
    df = pd.DataFrame({"metric": [], "value": [], "step": []})
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]
    for tag in tags:
        event_list = event_acc.Scalars(tag)
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))
        r = {"metric": [tag] * len(step), "value": values, "step": step}
        r = pd.DataFrame(r)
        df = pd.concat([df, r])
    return df


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
    agg = defaultdict(list)
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
                    agg[distance[i, j]].append(correlations[b, i, j].item())
    return agg
