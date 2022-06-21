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
from tqdm.auto import tqdm

from topography import TopographicModel, models
from topography.utils import plot as topo_plot


@dataclasses.dataclass(frozen=True)
class CIFAR10TopographicPlotConfig:
    log: str  # Output directory
    data: str  # Data directory
    batch_size: int  # Batch size
    model: str  # Base model
    normalization: List = dataclasses.field(
        default_factory=lambda: [
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ]
    )  # CIFAR image normalization


def main(config: CIFAR10TopographicPlotConfig):
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
    model = TopographicModel(getattr(models, config.model)()).to(device)

    state_dict_path = sorted(logdir.joinpath("checkpoints").glob("*.model"))[-1]
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    for layer in tqdm(model._conv_layer_names):
        agg = topo_plot.aggregate_correlation(
            model, single_loader, layer, device
        )
        fig, ax = topo_plot.plot_aggregated_correlations(
            agg, marker="o", ecolor="g"
        )
        ax.legend()
        ax.set_title(layer)
        fig.savefig(output.joinpath(f"correlation_{layer}.pdf"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", type=str, help="Output directory.", required=True
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

    config = CIFAR10TopographicPlotConfig(
        log=args.log,
        data=config_json["data"],
        batch_size=config_json["batch_size"],
        model=config_json["model"],
        normalization=config_json["normalization"],
    )
    main(config)
