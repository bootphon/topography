"""Correlation matrix accross the training epochs.

This scripts works given a certain layer, channel and input image.
For each saved checkpoint, it plots a comparison
between the cosine similarity of the output of the considered channel
and the outputs of the other channels, and the target inverse distance we
wish to fit. It also plots the resulting loss.
This is done using every model checkpoint we saved: we are then able
to see how the topography is learned for this image, layer and channel.
For now it only works on CIFAR.
A possible extension would be to look at the mean correlation
accross the entire dataset, and not just at one image.
"""
import argparse
import dataclasses
import json
import math
import random
import subprocess
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from topography import TopographicModel, models
from topography.core.loss import _channel_correlation

DIMENSION: int = 2


@dataclasses.dataclass(frozen=True)
class EpochsCorrelationConfig:
    """Dataclass to store the whole configuration used."""

    logdir: Path  # Output directory.
    data: str  # Data directory.

    idx: int  # Image index in the training set.
    layer_name: str  # Name of the considered layer
    channel_idx: int  # Index of the considered channel

    model: str  # Model to use.
    num_classes: int  # Number of CIFAR classes, either 10 or 100.
    norm: str  # Which norm between positions to use.
    normalization: List  # CIFAR image normalization.

    framerate: int = 10  # Framerate for the recap video


def main(config: EpochsCorrelationConfig) -> None:
    plotdir = config.logdir / "plot" / "correlation_matrix" / "epochs"
    plotdir.mkdir(exist_ok=True, parents=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*config.normalization)]
    )
    dataset = (
        datasets.CIFAR10 if config.num_classes == 10 else datasets.CIFAR100
    )
    img = dataset(
        root=config.data, train=True, download=False, transform=transform
    )[config.idx][0]
    name = config.layer_name

    base_model = getattr(models, config.model)(num_classes=config.num_classes)
    model = TopographicModel(base_model, dimension=DIMENSION, norm=config.norm)
    state_dicts = sorted(config.logdir.joinpath("checkpoints").glob("*.model"))

    for idx, state_dict in enumerate(tqdm(state_dicts)):
        model.load_state_dict(torch.load(state_dict, map_location="cpu"))
        model.eval()

        model(img.unsqueeze(0))
        activation = model.activations[name]
        inv_dist = model.inverse_distance[name]
        correlation = _channel_correlation(activation, 1e-8).detach()
        correlation = (correlation[0] + correlation[0].T).fill_diagonal_(1)
        inv_dist = (inv_dist + inv_dist.T).fill_diagonal_(1)

        num_channels = correlation.shape[0]
        num_axis = int(math.ceil(num_channels ** (1 / DIMENSION)))

        norm_corr = Normalize(vmin=-1, vmax=1)

        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5), facecolor="white"
        )
        cax1 = fig.add_axes([0.05, 0.15, 0.02, 0.7])
        cax2 = fig.add_axes([0.93, 0.15, 0.02, 0.7])

        channel_corr = torch.zeros(num_axis * num_axis)
        channel_corr[:num_channels] = correlation[config.channel_idx]
        channel_corr = channel_corr.reshape(num_axis, num_axis)

        channel_inv_dist = torch.zeros(num_axis * num_axis)
        channel_inv_dist[:num_channels] = inv_dist[config.channel_idx]
        channel_inv_dist = channel_inv_dist.reshape(num_axis, num_axis)

        ax[0].imshow(channel_corr, norm=norm_corr, cmap="twilight")
        ax[0].set_title(r"$Corr$")
        ax[0].axis("off")

        im1 = ax[1].imshow(channel_inv_dist, norm=norm_corr, cmap="twilight")
        ax[1].set_title(r"$\frac{1}{d+1}$")
        ax[1].axis("off")
        fig.colorbar(im1, cax=cax1)

        loss = (channel_corr - channel_inv_dist) ** 2
        norm_loss = Normalize(vmin=loss.min(), vmax=loss.max())

        im2 = ax[2].imshow(loss, cmap="plasma", norm=norm_loss)
        ax[2].set_title(r"$Loss = (Corr - \frac{1}{d+1})^2$")
        ax[2].axis("off")
        fig.colorbar(im2, cax=cax2)

        fig.suptitle(
            f"Epoch {int(state_dict.stem)} ({name}"
            f" , channel {config.channel_idx})"
        )
        fig.savefig(plotdir / f"corr_matrix_{idx:04d}.png")
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
    parser.add_argument(
        "--channel_idx",
        type=int,
        help="Image index in the training set.",
        required=True,
    )
    parser.add_argument("--framerate", type=int, default=10, help="Framerate.")
    args = parser.parse_args()

    with open(
        Path(args.log) / "environment/config.json", "r", encoding="utf-8"
    ) as file:
        config_json = json.load(file)

    random.seed(config_json["seed"])
    np.random.seed(config_json["seed"])
    torch.manual_seed(config_json["seed"])
    torch.cuda.manual_seed_all(config_json["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = EpochsCorrelationConfig(
        logdir=Path(args.log).resolve(),
        data=config_json["data"],
        idx=args.idx,
        layer_name=args.layer_name,
        channel_idx=args.channel_idx,
        model=config_json["model"],
        num_classes=config_json["num_classes"],
        norm=config_json["norm"],
        normalization=config_json["normalization"],
        framerate=args.framerate,
    )
    main(config)
