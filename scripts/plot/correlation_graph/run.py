"""Script to plot the pairwise correlation between channels as a function of
the distance, for each topographic layer for a given model.
"""
import argparse
import dataclasses
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from topography import TopographicModel, models
from topography.core.loss import _channel_correlation
from topography.utils.data import BirdDCASE, SpeechCommands


@dataclasses.dataclass(frozen=True)
class CorrelationPlotConfig:
    """Pairwise correlation plot configuration"""

    log: str  # Output directory
    data: str  # Data directory
    dataset: str  # Dataset used
    subset: str  # Subset to consider
    batch_size: int  # Batch size
    model: str  # Base model
    num_classes: int  # Number of classes in CIFAR
    normalization: List  # CIFAR image normalization
    dimension: int  # Dimension of the positions.
    norm: str  # Which norm between positions to use.
    position_scheme: str  # Position scheme (hypercube, nested or hypersphere).
    overwrite: bool = False  # Whether to overwrite existing files.

    fig_kw: Dict = dataclasses.field(default_factory=dict)  # Figure kwargs
    scatter_kw: Dict = dataclasses.field(default_factory=dict)  # Plot kwargs
    ref_kw: Dict = dataclasses.field(default_factory=dict)  # Ref kwargs


def plot_correlations(
    correlations: torch.Tensor,
    inverse_distance: torch.Tensor,
    fig_kw: Dict,
    scatter_kw: Dict,
    ref_kw: Dict,
) -> Tuple:
    """Plot the pairwise correlations between channels for a given layer.

    Parameters
    ----------
    correlations : torch.Tensor
        Pairwise correlations between channels, averaged across the entire
        dataset. Tensor of shape (`num_channels`, `num_channels`), upper
        triangular with a diagonal full of 0.
    inverse_distance : torch.Tensor
        Inverse distance between channels.
        Tensor of shape (`num_channels`, `num_channels`), upper
        triangular with a diagonal full of 0.
    fig_kw : Dict
        Figure configuration.
    scatter_kw : Dict
        Configuration for the scatter plot of the paiwise correlations
        with plt.scatter.
    ref_kw : Dict
        Configuration for plotting the target function with plt.plot.

    Returns
    -------
    Tuple
        Tuple of matplotlib Figure and Axes.
    """
    row_idx, col_idx = torch.triu_indices(*inverse_distance.shape, offset=1)
    distance = 1 / inverse_distance[row_idx, col_idx] - 1
    ref = np.linspace(0, max(distance))

    fig, ax = plt.subplots(**fig_kw)
    ax.plot(ref, 1 / (ref + 1), **ref_kw)
    ax.scatter(distance, correlations[row_idx, col_idx], **scatter_kw)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Correlation")
    return fig, ax


def aggregate_correlations(
    model: TopographicModel, dataloader: DataLoader, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Aggregate the pairwise correlations between channels
    in each topographic layer across the entire dataset.

    Parameters
    ----------
    model : TopographicModel
        Topographic model under scrutiny.
    dataloader : DataLoader
        Dataloader.
    device : torch.device
        PyTorch device.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionnary of pairwise correlations averaged across the
        full dataset. Maps a given layer name to an upper triangular tensor
        of shape (`num_channels`, `num_channels`) with a diagonal of 0,
        containing the pairwise correlations between the `num_channels`
        channels of this layer.
    """
    correlations = defaultdict(list)
    with torch.no_grad():
        for batch, _ in tqdm(dataloader, "Correlations"):
            model(batch.to(device))
            for layer, activ in model.activations.items():
                correlations[layer].append(
                    _channel_correlation(activ, 1e-8).cpu()
                )
    return {
        layer: torch.vstack(corr_list).mean(axis=0)
        for layer, corr_list in correlations.items()
    }


def main(config: CorrelationPlotConfig) -> None:
    """Computes and plots the pairwise correlation between channels.

    Parameters
    ----------
    config : CorrelationPlotConfig
        Full configuration.

    Raises
    ------
    ValueError
        If the specified dataset is incorrect.
    """
    logdir = Path(config.log)
    plotdir = logdir.joinpath("plot")
    plotdir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.dataset.startswith("cifar"):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*config.normalization)]
        )
        num_classes, in_channels = int(config.dataset.removeprefix("cifar")), 3
        dataset = (
            datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100
        )(
            root=config.data,
            train=config.subset == "training",
            download=False,
            transform=transform,
        )
    elif config.dataset == "speechcommands":
        num_classes, in_channels = 35, 1
        dataset = SpeechCommands(config.data, subset=config.subset)
    elif config.dataset == "birddcase":
        num_classes, in_channels = 2, 1
        dataset = BirdDCASE(config.data, subset=config.subset)
    else:
        raise ValueError(f"Wrong dataset {config.dataset}")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size
    )

    base_model = getattr(models, config.model)(
        num_classes=config.num_classes, in_channels=in_channels
    )
    model = TopographicModel(
        base_model,
        dimension=config.dimension,
        norm=config.norm,
        position_scheme=config.position_scheme,
        topographic_layer_names=models.topographic_layer_names(config.model),
    ).to(device)

    state_dict_path = sorted(logdir.joinpath("checkpoints").glob("*.model"))[-1]
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()

    correlations = aggregate_correlations(model, dataloader, device)
    for layer in model.topographic_layer_names:
        out_fig = plotdir.joinpath(f"correlation_{layer}.pdf")
        if not out_fig.exists() or config.overwrite:
            fig, ax = plot_correlations(
                correlations[layer],
                model.inverse_distance[layer].cpu(),
                fig_kw=config.fig_kw,
                ref_kw=config.ref_kw,
                scatter_kw=config.scatter_kw,
            )
            ax.legend()
            ax.set_title(layer)
            fig.savefig(out_fig)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", type=str, help="Output directory.", required=True
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="training",
        help="Subset of the dataset to consider.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files or not.",
    )
    parser.add_argument("--style", type=str, help="Path to mplstyle file.")
    args = parser.parse_args()

    if args.style is not None:
        plt.style.use(args.style)

    with open(
        Path(args.log).joinpath("environment/config.json"),
        "r",
        encoding="utf-8",
    ) as file:
        config_json = json.load(file)

    random.seed(config_json["seed"])
    np.random.seed(config_json["seed"])
    torch.manual_seed(config_json["seed"])
    torch.cuda.manual_seed_all(config_json["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = CorrelationPlotConfig(
        log=args.log,
        data=config_json["data"],
        subset=args.subset,
        dataset=config_json["dataset"],
        batch_size=config_json["batch_size"],
        model=config_json["model"],
        num_classes=config_json["num_classes"],
        normalization=config_json["normalization"],
        dimension=config_json["dimension"],
        norm=config_json["norm"],
        position_scheme=config_json["position_scheme"],
        overwrite=args.overwrite,
        fig_kw={"figsize": (12, 8)},
        ref_kw={"label": r"$\frac{1}{x+1}$"},
        scatter_kw={"alpha": 0.1, "s": 0.5, "c": "k"},
    )
    main(config)
