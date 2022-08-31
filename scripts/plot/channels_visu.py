"""Script to visualize channels by finding the highest activating images,
using the Lucent library.
"""
import argparse
import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lucent.optvis import param, render
from PIL import Image
from tqdm.auto import tqdm

from topography import TopographicModel, models
from topography.core.distance import hypercube

# Cf https://distill.pub/2017/feature-visualization/#preconditioning
# in order to understand these parameters and how alternative
# parametrizations can produce better results.
FFT: bool = True
DECORRELATE: bool = True


def run(model: TopographicModel, plotdir: Path, in_channels: int) -> None:
    """Find the highest acitvating images for every channel of every
    Conv2d layer with topography.

    Parameters
    ----------
    model : TopographicModel
        The topographic model.
    plotdir : Path
        The directory where to save the resulting images.
    in_channels: int
        Number of channels in the image (either 1 or 3).
    """
    plotdir.mkdir(exist_ok=True, parents=True)
    param_f = lambda: param.image(
        128,
        channels=in_channels,
        fft=FFT,
        decorrelate=DECORRELATE,
    )
    for layer_name, inv_dist in model.inverse_distance.items():
        out_channels = inv_dist.shape[0]
        layer = layer_name.replace(".", "_")
        for channel in tqdm(range(out_channels), desc=f"Render {layer}"):
            render.render_vis(
                model,
                f"model_{layer}:{channel}",
                param_f,
                show_image=False,
                save_image=True,
                image_name=plotdir / f"{layer}-{channel}.png",
                progress=False,
            )


def process(plotdir: Path) -> None:
    """Arange the resulting images according to the 2D grid topography,
    for every layer.

    Parameters
    ----------
    plotdir : Path
        Directory where the resulting images were saved.
        The final visualization will be saved in `plotdir.parent`.
    """
    imgs = defaultdict(dict)

    for path in plotdir.glob("*.png"):
        layer_name, channel = path.stem.split("-")
        imgs[layer_name][int(channel)] = deepcopy(Image.open(path))

    for current_layer in tqdm(imgs.keys(), desc="Plot grid"):
        num_channels = len(imgs[current_layer])

        positions = hypercube(
            num_points=num_channels,
            dimension=2,
            integer_positions=True,
        )
        num_axis = positions.max().item() + 1

        fig, ax = plt.subplots(
            nrows=num_axis,
            ncols=num_axis,
            figsize=(2 * num_axis, 2 * num_axis),
        )

        for channel, (i, j) in enumerate(positions):
            ax[num_axis - 1 - j, i].imshow(imgs[current_layer][channel])
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].axis("off")

        fig.suptitle(current_layer)
        fig.savefig(plotdir.parent / f"{current_layer}.pdf")
        fig.savefig(plotdir.parent / f"{current_layer}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", type=str, required=True, help="Logging directory"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    logdir = Path(args.log).resolve()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(logdir / "environment/config.json", "r") as f:
        config = json.load(f)

    dataset = config["dataset"]
    if dataset.startswith("cifar"):
        num_classes, in_channels = int(dataset.removeprefix("cifar")), 3
    elif dataset == "speechcommands":
        num_classes, in_channels = 35, 1
    elif dataset == "birddcase":
        num_classes, in_channels = 2, 1
    else:
        raise ValueError(f"Wrong dataset {dataset}")

    model_name = config["model"]
    base_model = getattr(models, model_name)(
        num_classes=num_classes, in_channels=in_channels
    )
    state_dict = torch.load(
        sorted((logdir / "checkpoints").glob("*.model"))[-1],
        map_location=device,
    )
    topo_names = models.topographic_layer_names(model_name)
    if config["topographic"]:
        if config["dimension"] != 2 or config["position_scheme"] != "hypercube":
            raise ValueError(
                "If the model is topographic, it must be using"
                + " a 2D regular grid topography."
            )
        model = (
            TopographicModel(base_model, topographic_layer_names=topo_names)
            .eval()
            .to(device)
        )
        model.load_state_dict(state_dict)
    else:
        base_model.load_state_dict(state_dict)
        model = TopographicModel(base_model, topo_names).eval().to(device)

    run(model, logdir / "plot/lucent", in_channels)
    process(logdir / "plot/lucent")
