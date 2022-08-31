"""Script to compute the category selectivity.
"""
import argparse
import dataclasses
import itertools
import json
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from topography import TopographicModel, models
from topography.utils import AverageMeter
from topography.utils.data import BirdDCASE, SpeechCommands

Stats = Dict[Tuple[str, int], AverageMeter]

DIMENSION: int = 2
POSITION_SCHEME: str = "hypercube"


@dataclasses.dataclass(frozen=True)
class CategorySelectivityConfig:
    """Category selectivity configuration"""

    plotdir: Path  # Plot directory.
    data: str  # Data directory.
    dataset: str  # Dataset used.
    subset: str  # Subset to consider.
    model: str  # Model to use.
    classes: Optional[List[int]] = None  # Subset of class index to consider.
    normalization: Optional[List] = None  # CIFAR image normalization.
    batch_size: int = 8  # Batch size.


def compute_stats(
    model: TopographicModel,
    dataloader: torch.utils.data.DataLoader,
    classes: List[int],
) -> Tuple[Stats, Stats]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    sums_meters = {
        (layer_name, class_idx): AverageMeter(f"{layer_name},{class_idx}")
        for class_idx in classes
        for layer_name in model.inverse_distance.keys()
    }
    squares_meters = {
        (layer_name, class_idx): AverageMeter(f"{layer_name},{class_idx}")
        for class_idx in classes
        for layer_name in model.inverse_distance.keys()
    }
    for data, labels in tqdm(dataloader):
        if not any(label.item() in classes for label in labels):
            continue
        model(data.to(device))
        for idx, label in enumerate(labels):
            label = label.item()
            if label not in classes:
                continue
            for layer_name, activation in model.activations.items():
                activation = activation.detach().cpu()
                sums_meters[(layer_name, label)].update(activation[idx])
                squares_meters[(layer_name, label)].update(activation[idx] ** 2)
    mean = {key: meter.avg for key, meter in sums_meters.items()}
    var = {
        key: meter.avg - mean[key] ** 2 for key, meter in squares_meters.items()
    }

    return mean, var


def category_selectivity(
    mean: Stats,
    var: Stats,
) -> Dict[Tuple[str, int, int], np.ndarray]:
    layer_names, classes = zip(*mean.keys())
    layer_names, classes = set(layer_names), set(classes)
    all_selectivity = {}
    for layer_name, i, j in itertools.product(layer_names, classes, classes):
        for j in classes:
            if i == j:
                continue
            selectivity = (
                (mean[(layer_name, i)] - mean[(layer_name, j)])
                / torch.sqrt(
                    0.5 * (var[(layer_name, i)] + var[(layer_name, j)])
                )
            ).mean(axis=(1, 2))
            num_axis = int(math.ceil(len(selectivity) ** (1 / DIMENSION)))
            d_full = torch.zeros(num_axis * num_axis)
            d_full[: len(selectivity)] = selectivity
            img = np.ma.array(
                d_full.numpy(),
                mask=np.arange(len(d_full)) >= len(selectivity),
            )
            all_selectivity[(layer_name, i, j)] = img.reshape(
                num_axis, num_axis
            )
    return all_selectivity


def main(config: CategorySelectivityConfig) -> None:
    """_summary_

    Parameters
    ----------
    config : CategorySelectivityConfig
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(Path(__file__).parent / "classes.json", "r") as file:
        class_names = json.load(file)[config.dataset]

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
        dimension=DIMENSION,
        position_scheme=POSITION_SCHEME,
        topographic_layer_names=models.topographic_layer_names(config.model),
    )

    state_dict = sorted(logdir.joinpath("checkpoints").glob("*.model"))[-1]
    model.load_state_dict(torch.load(state_dict, map_location=device))
    model.eval()

    mean, var = compute_stats(model, dataloader, config.classes)
    with open(config.plotdir / "stats.pkl", "wb") as file:
        pickle.dump({"mean": mean, "var": var}, file)

    selectivity = category_selectivity(mean, var)

    pdf = PdfPages(config.plotdir / "category_selectivity.pdf")
    num_classes = len(config.classes)
    for layer_name in tqdm(model.inverse_distance.keys()):
        keys = [key for key in selectivity.keys() if key[0] == layer_name]
        imgs = [selectivity[k] for k in keys]
        norm = Normalize(vmin=np.min(imgs), vmax=np.max(imgs))
        fig, ax = plt.subplots(num_classes, num_classes, figsize=(20, 20))
        k = 0
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    continue
                img_imshow = ax[i, j].imshow(imgs[k], norm=norm, origin="lower")
                ax[i, j].set_title(
                    f"{config.class_names[keys[k][1]]}, {config.class_names[keys[k][2]]}",
                    fontsize=10,
                )
                k += 1
        for i in range(num_classes):
            for j in range(num_classes):
                ax[i, j].axis("off")
        fig.colorbar(img_imshow, ax=ax.ravel())
        plt.suptitle(layer_name, fontsize=40)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
    pdf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=str,
        help="Output directory.",
        required=True,
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="training",
        help="Subset of the dataset to consider.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="List of classes to consider if provided.",
    )
    args = parser.parse_args()

    logdir = Path(args.log).resolve()
    with open(logdir / "environment/config.json", "r") as file:
        config_json = json.load(file)
    plotdir = logdir / "plot"
    plotdir.mkdir(exist_ok=True)

    random.seed(config_json["seed"])
    np.random.seed(config_json["seed"])
    torch.manual_seed(config_json["seed"])
    torch.cuda.manual_seed_all(config_json["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = CategorySelectivityConfig(
        plotdir=plotdir,
        data=config_json["data"],
        model=config_json["model"],
        dataset=config_json["dataset"],
        subset=args.subset,
        classes=args.classes,
        normalization=config_json.get("normalization"),
    )
    main(config)
