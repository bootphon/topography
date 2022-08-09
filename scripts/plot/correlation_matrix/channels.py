"""Correlation matrix accross the channels.

This scripts works given a certain checkpoint, layer and input image.
For each considered channel, it plots a comparison
between the cosine similarity of its output and the outputs of the
other channels, and the target inverse distance we wish to fit.
It also plots the resulting loss. This is done across every channel.
A possible extension would be to look at the mean correlation
across the entire dataset, and not just at one image.
"""
import argparse
import dataclasses
import json
import math
import random
import subprocess
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from topography import TopographicModel, models
from topography.core.loss import _channel_correlation
from topography.utils.data import BirdDCASE, SpeechCommands

DIMENSION: int = 2


@dataclasses.dataclass(frozen=True)
class ChannelsCorrelationConfig:
    """Dataclass to store the whole configuration used."""

    logdir: Path  # Output directory.
    data: str  # Data directory.
    dataset: str  # Dataset used

    idx: int  # Image index in the training set.
    layer_name: str  # Name of the considered layer
    state_dict: Path  # State dict path

    model: str  # Model to use.
    norm: str  # Which norm between positions to use.
    normalization: Optional[List] = None  # CIFAR image normalization.

    framerate: int = 10  # Framerate for the recap video


def main(config: ChannelsCorrelationConfig) -> None:
    plotdir = config.logdir / "plot" / "correlation_matrix" / "channels"
    plotdir.mkdir(exist_ok=True, parents=True)

    if config.dataset.startswith("cifar"):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*config.normalization)]
        )
        num_classes = int(config.dataset.removeprefix("cifar"))
        in_channels = 3
        dataset = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100
        inp = dataset(
            root=config.data, train=True, download=False, transform=transform
        )[config.idx][0]

    elif config.dataset == "speechcommands":
        num_classes, in_channels = 35, 1
        inp = SpeechCommands(config.data, subset="training")[config.idx][0]

    elif config.dataset == "birddcase":
        num_classes, in_channels = 2, 1
        inp = BirdDCASE(config.data, subset="training")[config.idx][0]

    else:
        raise ValueError(f"Wrong dataset {config.dataset}")

    name = config.layer_name

    state_dict = torch.load(config.state_dict, map_location="cpu")
    base_model = getattr(models, config.model)(
        num_classes=num_classes, in_channels=in_channels
    )
    model = TopographicModel(
        base_model,
        dimension=DIMENSION,
        norm=config.norm,
        topographic_layer_names=models.topographic_layer_names(config.model),
    )
    model.load_state_dict(state_dict)
    model.eval()

    model(inp.unsqueeze(0))
    activation, inv_dist = model.activations[name], model.inverse_distance[name]
    correlation = _channel_correlation(activation, 1e-8).detach()
    correlation = (correlation[0] + correlation[0].T).fill_diagonal_(1)
    inv_dist = (inv_dist + inv_dist.T).fill_diagonal_(1)

    num_channels = correlation.shape[0]
    num_axis = int(math.ceil(num_channels ** (1 / DIMENSION)))

    loss = (correlation - inv_dist) ** 2
    norm_corr = Normalize(vmin=inv_dist.min(), vmax=inv_dist.max())
    norm_loss = Normalize(vmin=loss.min(), vmax=loss.max())

    for channel_idx in tqdm(range(num_channels)):
        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5), facecolor="white"
        )
        cax1 = fig.add_axes([0.05, 0.15, 0.02, 0.7])
        cax2 = fig.add_axes([0.93, 0.15, 0.02, 0.7])

        channel_corr = torch.zeros(num_axis * num_axis)
        channel_corr[:num_channels] = correlation[channel_idx]
        channel_corr = channel_corr.reshape(num_axis, num_axis)

        channel_inv_dist = torch.zeros(num_axis * num_axis)
        channel_inv_dist[:num_channels] = inv_dist[channel_idx]
        channel_inv_dist = channel_inv_dist.reshape(num_axis, num_axis)

        ax[0].imshow(channel_corr, norm=norm_corr)
        ax[0].set_title(r"$Corr$")
        ax[0].axis("off")

        im1 = ax[1].imshow(channel_inv_dist, norm=norm_corr)
        ax[1].set_title(r"$\frac{1}{d+1}$")
        ax[1].axis("off")
        fig.colorbar(im1, cax=cax1)

        loss = (channel_corr - channel_inv_dist) ** 2
        im2 = ax[2].imshow(loss, cmap="plasma", norm=norm_loss)
        ax[2].set_title(r"$Loss = (Corr - \frac{1}{d+1})^2$")
        ax[2].axis("off")
        fig.colorbar(im2, cax=cax2)

        fig.suptitle(f"Layer {name}, channel {channel_idx}.")
        fig.savefig(plotdir / f"corr_matrix_{channel_idx:04d}.png")
        plt.close()

    cmd = (
        f"ffmpeg -y -framerate {config.framerate}"
        f" -i {plotdir}/corr_matrix_%04d.png {plotdir}/output.mp4"
    )
    try:
        subprocess.run(cmd.split(), check=True)
    except FileNotFoundError:
        print("ffmpeg not found, cannot create the video.")
    except subprocess.CalledProcessError as error:
        print(f"ffmpeg command failed: {str(error)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=str,
        help="Output directory. It is structured similarly as the directories"
        + "created when training a model."
        + "It should contain 'checkpoints' and 'environment/config.json'.",
        required=True,
    )
    parser.add_argument(
        "--layer_name", type=str, help="Layer name.", required=True
    )
    parser.add_argument(
        "--idx",
        type=int,
        help="Image index in the training set.",
        required=True,
    )
    parser.add_argument("--framerate", type=int, default=10, help="Framerate.")
    parser.add_argument("--state_dict", type=str, help="Model state dict.")
    args = parser.parse_args()

    with open(
        Path(args.log) / "environment/config.json", "r", encoding="utf-8"
    ) as file:
        config_json = json.load(file)

    if "dimension" not in config_json or config_json["dimension"] != 2:
        raise ValueError("Dimension must be 2.")

    random.seed(config_json["seed"])
    np.random.seed(config_json["seed"])
    torch.manual_seed(config_json["seed"])
    torch.cuda.manual_seed_all(config_json["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logdir = Path(args.log).resolve()
    if args.state_dict is None:
        state_dict = sorted(logdir.joinpath("checkpoints").glob("*.model"))[-1]
    else:
        state_dict = logdir / "checkpoints" / args.state_dict
        if not state_dict.is_file():
            raise ValueError(f"Invalid state dict {args.state_dict}.")

    config = ChannelsCorrelationConfig(
        logdir=logdir,
        data=config_json["data"],
        dataset=config_json["dataset"],
        idx=args.idx,
        layer_name=args.layer_name,
        model=config_json["model"],
        norm=config_json["norm"],
        normalization=config_json.get("normalization"),
        state_dict=state_dict,
        framerate=args.framerate,
    )
    main(config)
