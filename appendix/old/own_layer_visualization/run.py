import argparse
import dataclasses
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch import nn
from tqdm.auto import tqdm

from topography import TopographicModel, models
from topography.core.distance import hypercube

from visualization import VisualizationLayerCNN, Normalization


@dataclasses.dataclass(frozen=True)
class LayersVisuConfig:
    log: str  # Output directory
    model: str  # Base model
    num_classes: int  # Number of classes in CIFAR
    normalization: Normalization  # CIFAR image normalization

    topographic: bool
    dimension: Optional[int] = None  # Dimension of the positions.
    norm: Optional[str] = None  # Which norm between positions to use.

    shape: Tuple = (3, 32, 32)  # Image shape
    n_iter: int = 100  # Number of iterations
    overwrite: bool = False  # Whether to overwrite existing files.


def main(config):
    logdir = Path(config.log)
    output = logdir / "plot" / "visu_layers"
    output.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = getattr(models, config.model)(num_classes=config.num_classes)
    if config.topographic:
        model = TopographicModel(
            base_model, dimension=config.dimension, norm=config.norm
        ).to(device)
    else:
        model = base_model.to(device)
    state_dict_path = sorted(logdir.joinpath("checkpoints").glob("*.model"))[-1]
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()

    visualizer = VisualizationLayerCNN(model, config.normalization)

    layers_channels = [
        (name, module.out_channels)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]

    imgs = {}
    out_single_img = output / "imgs"
    out_single_img.mkdir(exist_ok=True)
    for layer_name, num_channels in tqdm(layers_channels):
        for channel in range(num_channels):
            img = visualizer(
                layer_name, channel, config.n_iter, config.shape, device
            )
            img.save(
                out_single_img
                / f"{layer_name.replace('.', '-')}_{channel}_color.png"
            )
            img.convert("L").save(
                out_single_img
                / f"{layer_name.replace('.', '-')}_{channel}_gray.png"
            )
            imgs[(layer_name, channel)] = img

    pdf_color = PdfPages(output / "layers_color.pdf")
    pdf_gray = PdfPages(output / "layers_gray.pdf")

    for layer_name, num_channels in tqdm(layers_channels):
        positions = hypercube(
            num_points=num_channels,
            dimension=config.dimension,
            integer_positions=True,
        )
        num_axis = positions.max().item() + 1

        if config.dimension == 2:
            fig_color, ax_color = plt.subplots(
                nrows=num_axis,
                ncols=num_axis,
                figsize=(2 * num_axis, 2 * num_axis),
            )
            fig_gray, ax_gray = plt.subplots(
                nrows=num_axis,
                ncols=num_axis,
                figsize=(2 * num_axis, 2 * num_axis),
            )
            for channel, (i, j) in enumerate(positions):
                ax_color[num_axis - 1 - j, i].imshow(
                    imgs[(layer_name, channel)], vmin=0, vmax=255
                )
                ax_gray[num_axis - 1 - j, i].imshow(
                    imgs[(layer_name, channel)].convert("L"),
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
            for k in range(ax_color.shape[0]):
                for l in range(ax_color.shape[1]):
                    ax_color[k, l].axis("off")
                    ax_gray[k, l].axis("off")

        elif config.dimension == 1:
            fig_color, ax_color = plt.subplots(
                ncols=num_axis, figsize=(2 * num_axis, 6)
            )
            fig_gray, ax_gray = plt.subplots(
                ncols=num_axis, figsize=(2 * num_axis, 6)
            )
            for channel, k in enumerate(positions):
                ax_color[k].imshow(
                    imgs[(layer_name, channel)], vmin=0, vmax=255
                )
                ax_gray[k].imshow(
                    imgs[(layer_name, channel)].convert("L"),
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
            for l in range(len(ax_color)):
                ax_color[l].axis("off")
                ax_gray[l].axis("off")

        else:
            raise ValueError(
                f"Dimension must be 1 or 2, not {config.dimension} in order to"
                + "plot the map of maximum activating images."
            )
        fig_color.suptitle(layer_name, fontsize=30)
        fig_gray.suptitle(layer_name, fontsize=30)
        pdf_color.savefig(fig_color, bbox_inches="tight")
        pdf_gray.savefig(fig_gray, bbox_inches="tight")
        plt.close()
    pdf_color.close()
    pdf_gray.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files or not.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=100,
        help="Number of optimizer iterations.",
    )
    args = parser.parse_args()
    with open(Path(args.log) / "environment/config.json", "r") as file:
        config_json = json.load(file)

    random.seed(config_json["seed"])
    np.random.seed(config_json["seed"])
    torch.manual_seed(config_json["seed"])
    torch.cuda.manual_seed_all(config_json["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = LayersVisuConfig(
        log=args.log,
        model=config_json["model"],
        num_classes=config_json["num_classes"],
        normalization=config_json["normalization"],
        topographic=config_json["topographic"],
        dimension=config_json["dimension"],
        norm=config_json["norm"],
        n_iter=args.n_iter,
        overwrite=args.overwrite,
    )
    main(config)
