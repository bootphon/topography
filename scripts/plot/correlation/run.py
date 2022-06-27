"""Script to plot the correlation for each layer of each model."""
import argparse
import dataclasses
import json
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms

from topography import TopographicModel, models
from topography.utils import plot as topo_plot


@dataclasses.dataclass(frozen=True)
class CIFARTopographicPlotConfig:
    log: str  # Output directory
    data: str  # Data directory
    batch_size: int  # Batch size
    model: str  # Base model
    num_classes: int  # Number of classes in CIFAR
    normalization: List  # CIFAR image normalization
    dimension: int  # Dimension of the positions.
    norm: str  # Which norm between positions to use.
    overwrite: bool = False  # Whether to overwrite existing files.


def main(config: CIFARTopographicPlotConfig):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*config.normalization),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=config.data,
        train=True,
        download=False,
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    single_loader = [next(iter(dataloader))]

    logdir = Path(config.log)
    output = logdir.joinpath("plot")
    output.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = getattr(models, config.model)(num_classes=config.num_classes)
    model = TopographicModel(
        base_model, dimension=config.dimension, norm=config.norm
    ).to(device)

    state_dict_path = sorted(logdir.joinpath("checkpoints").glob("*.model"))[-1]
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    for idx, layer in enumerate(model._conv_layer_names):
        print(f"Layer {layer}, {idx+1}/{len(model._conv_layer_names)}")
        out_fig = output.joinpath(f"correlation_{layer}.pdf")
        if not out_fig.exists() or config.overwrite:
            agg = topo_plot.aggregate_correlation(
                model, single_loader, layer, device
            )
            fig, ax = topo_plot.plot_aggregated_correlations(
                agg, marker="o", ecolor="g"
            )
            ax.legend()
            ax.set_title(layer)
            fig.savefig(out_fig)
            fig.savefig(out_fig.with_suffix(".png"))
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", type=str, help="Output directory.", required=True
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files or not.",
    )
    args = parser.parse_args()

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

    if "norm" not in config_json:
        config_json["norm"] = "euclidean"

    config = CIFARTopographicPlotConfig(
        log=args.log,
        data=config_json["data"],
        batch_size=config_json["batch_size"],
        model=config_json["model"],
        num_classes=config_json["num_classes"],
        normalization=config_json["normalization"],
        dimension=config_json["dimension"],
        norm=config_json["norm"],
        overwrite=args.overwrite,
    )
    main(config)
