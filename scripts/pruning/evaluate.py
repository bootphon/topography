"""Script to evaluate models after having pruned the topographic layers."""
import argparse
import dataclasses
import json
import random
from operator import attrgetter
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune
from torch.nn.utils.prune import LnStructured, RandomStructured
from torchvision import datasets, transforms
from tqdm import tqdm

from topography import TopographicModel, models
from topography.training import Writer, evaluate
from topography.utils import topographic_pruning
from topography.utils.data import BirdDCASE, SpeechCommands

SEED: int = 0
BATCH_SIZE: int = 128
SUBSET: str = "testing"


def topographic_pruning(
    model: TopographicModel,
    pruning_method: prune.BasePruningMethod,
    layer_name: str,
    **kwargs,
) -> TopographicModel:
    """Prune a given topographic layer of a model according
    to a pruning method.

    Parameters
    ----------
    model : TopographicModel
        Model to prune.
    pruning_method : prune.BasePruningMethod
        PyTorch pruning method to use.
    layer_name : str
        Name of the topographic layer to prune.
    **kwargs:
        Arguments to the `pruning_method`.

    Returns
    -------
    TopographicModel
        Pruned model.

    Raises
    ------
    ValueError
        If the target layer is not a topographic layer.
    """
    if layer_name not in model.topographic_layer_names:
        raise ValueError(f"Layer {layer_name} is not a topographic layer.")
    module = attrgetter(layer_name)(model.model)
    pruning_method.apply(module, "weight", **kwargs)
    if hasattr(module, "bias") and module.bias is not None:
        mask = module.weight_mask.mean(axis=(1, 2, 3))
        prune.CustomFromMask.apply(module, "bias", mask=mask)
    return model


@dataclasses.dataclass(frozen=True)
class PruningConfig:
    """Pruning configuration"""

    logdir: str  # Logging directory.
    data: str  # Data directory.
    dataset: str  # Dataset used.
    model: str  # Model to use.
    seed: int  # Random seed used to train the model

    proportion: float  # Proportion of the channels of each topo layer to prune.
    mode: str  # Pruning method
    ln_dimension: int  # Dimension used in the LnStructured pruning.
    subset: str  # Subset of the dataset considered for evaluation.
    batch_size: int  # Batch size.

    topographic: bool  # Whether to use a topographic model or not
    dimension: Optional[int] = None  # Dimension of the positions.
    norm: Optional[str] = None  # Which norm between positions to use.
    position_scheme: Optional[str] = None  # How to assign positions.
    normalization: Optional[List] = None  # CIFAR image normalization.


def main(
    config: PruningConfig,
    dataloader: torch.utils.data.DataLoader,
    criterion: Callable,
    device: torch.device,
    in_channels: int,
    num_classes: int,
) -> None:
    """Evaluates a model after having pruned the topographic layers.

    Parameters
    ----------
    config : PruningConfig
        Pruning configuration.
    dataloader : torch.utils.data.DataLoade
        Dataloader used
    criterion : Callable
        Classification loss.
    device : torch.device
        Torch device.
    in_channels : int
        Number of input channels (either 1 or 3)
    num_classes : int
        Number of classes
    """
    prunedir = Path(config.logdir) / "pruning_all"
    prunedir.mkdir(exist_ok=True)
    writer = Writer(prunedir)
    writer.log_config(dataclasses.asdict(config))

    # Model
    state_dict = sorted((Path(config.logdir) / "checkpoints").glob("*.model"))[
        -1
    ]  # Load the latest checkpoint
    base_model = getattr(models, config.model)(
        num_classes=num_classes, in_channels=in_channels
    )
    if not config.topographic:
        base_model.load_state_dict(torch.load(state_dict, map_location=device))
        dimension, position_scheme = 2, "hypercube"  # No influence
    else:
        dimension, position_scheme = config.dimension, config.position_scheme
    model = TopographicModel(
        base_model,
        dimension=dimension,
        position_scheme=position_scheme,
        topographic_layer_names=models.topographic_layer_names(config.model),
    ).to(device)
    if config.topographic:
        model.load_state_dict(torch.load(state_dict, map_location=device))
    model.eval()

    # Prune the model
    eval_mode = f"test_{config.mode}_all_{config.proportion:0.3f}"
    kwargs = {} if config.mode == "random" else {"n": config.ln_dimension}

    for layer_name in model.topographic_layer_names:
        model = topographic_pruning(
            model,
            RandomStructured if config.mode == "random" else LnStructured,
            layer_name=layer_name,
            amount=config.proportion,
            dim=0,
            **kwargs,
        )

    # Evaluation
    evaluate(
        model,
        dataloader,
        criterion,
        device,
        writer,
        is_pytorch_loss=True,
        mode=eval_mode,
    )
    writer.tensorboard.add_scalar(
        f"{eval_mode}/prop",
        config.proportion,
        writer._epochs[eval_mode],
    )

    writer.close()
    for logfile in prunedir.rglob("test_*.log"):
        logfile.unlink()


def job(
    logdir: Path,
    proportion: float,
    mode: str,
    seed: Optional[int] = None,
    ln_dimension: Optional[int] = None,
) -> None:
    if mode not in ["random", "weight"]:
        raise ValueError(f"Invalid pruning method {mode}")
    with open(logdir / "environment/config.json", "r") as file:
        config_json = json.load(file)

    seed = seed if seed is not None else config_json["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = PruningConfig(
        logdir=str(logdir),
        data=config_json["data"],
        dataset=config_json["dataset"],
        model=config_json["model"],
        topographic=config_json["topographic"],
        dimension=config_json["dimension"],
        position_scheme=config_json["position_scheme"],
        seed=config_json["seed"],
        proportion=proportion,
        mode=mode,
        ln_dimension=ln_dimension,
        subset=SUBSET,
        normalization=config_json.get("normalization"),
        batch_size=BATCH_SIZE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # Dataset
    if config.dataset.startswith("cifar"):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*config.normalization)]
        )
        num_classes, in_channels = int(config.dataset.removeprefix("cifar")), 3
        dataset = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100
        dataset = dataset(
            root=config.data,
            train=(config.subset == "training"),
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

    main(config, dataloader, criterion, device, in_channels, num_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir", type=str, help="Logging directory", required=True
    )
    args = parser.parse_args()

    logdir = Path(args.logdir)
    with open(logdir / "environment/config.json", "r") as file:
        config_json = json.load(file)

    model_name = config_json["model"]
    model = TopographicModel(
        getattr(models, model_name)(),
        topographic_layer_names=models.topographic_layer_names(model_name),
    )

    for proportion in tqdm(np.linspace(0, 1, 101)):
        job(logdir, proportion, "weight", None, 1)
        job(logdir, proportion, "weight", None, 2)
        job(logdir, proportion, "random", SEED, None)
