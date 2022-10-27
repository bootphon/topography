"""Script to train a model, topographic or not, on ImageNet."""
import argparse
import dataclasses
import random
import socket
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from topography import MetricOutput, TopographicLoss, TopographicModel, models
from topography.base import Metric, MetricOutput
from topography.training.training import accuracy
from topography.utils import AverageMeter, LinearWarmupCosineAnnealingLR

NUM_CLASSES = 1000
NORMALIZATION = [
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225),
]


def train_accelerate(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: Metric,
    accelerator: Accelerator,
) -> Dict[str, float]:
    """Training loop.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    dataloader : DataLoader
        Dataloader.
    optimizer : Optimizer
        Optimizer.
    criterion : Metric
        Loss function.
    accelerator : Accelerator
        Accelerator.
    """
    model.train()
    logs = defaultdict(lambda: AverageMeter(""))
    end = time.time()
    start_epoch = end

    for data, target in dataloader:
        # Measure data loading time
        logs["load-time"].update(time.time() - end)

        # Compute output
        output = model(data)
        loss = criterion(output, target)

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        accelerator.backward(loss.value)
        optimizer.step()

        # Compute accuracy
        acc = accuracy(output, target)

        # Measure accuracy and record loss
        logs["loss"].update(loss.value.item(), data.size(0))
        logs["acc"].update(acc.value, data.size(0))
        for name, value in {**loss.extras, **acc.extras}.items():
            logs[f"extras/{name}"].update(value, data.size(0))

        # Measure elapsed time
        logs["batch-time"].update(time.time() - end)
        end = time.time()
    total_epoch = start_epoch - end
    final_logs = {f"train/{k}": v.avg for k, v in logs.items()}
    final_logs["train/epoch-time"] = total_epoch
    return final_logs


def evaluate_accelerate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Metric,
    device: torch.device,
    mode: str = "test",
) -> Dict[str, float]:
    """Testing or validation loop.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    dataloader : DataLoader
        Dataloader.
    criterion : Callable
        Loss function.
    device : torch.device
        Device, either CPU or CUDA GPU.
    """
    model.eval()
    logs = defaultdict(lambda: AverageMeter(""))

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            acc = accuracy(output, target)
            logs["loss"].update(loss.value.item(), data.size(0))
            logs["acc"].update(acc.value, data.size(0))
            for name, value in {**loss.extras, **acc.extras}.items():
                logs[f"extras/{name}"].update(value, data.size(0))
    return {f"{mode}/{k}": v.avg for k, v in logs.items()}


@dataclasses.dataclass
class ImageNetConfig:
    """Dataclass to store the whole configuration used."""

    log: str  # Output directory.
    data: str  # Data directory.
    seed: int  # Random seed.

    model: str  # Model to use.
    topographic: bool  # Whether to train a topographic model or not.
    lambd: Optional[float] = None  # Weight of the topographic loss.
    dimension: Optional[int] = None  # Dimension of the positions.
    norm: Optional[str] = None  # Which norm between positions to use.

    epochs: int = 100  # Number of training epochs.
    val_size: int = 20_000  # Size of the validation set
    batch_size: int = 256  # Batch size.
    lr: float = 0.01  # Base learning rate.
    weight_decay: float = 0.01  # Weight decay.
    momentum: float = 0.9  # SGD momentum.
    resize_size: int = 256  # Resize size in random crop
    crop_size: int = 224  # Crop size
    horizontal_flip: float = 0.5  # Probability of horizontal flip in training.
    optimizer: str = "sgd"  # Optimizer.
    scheduler: str = "LinearWarmupCosineAnnealingLR"  # LR scheduler.
    warmup_epochs_prop: float = 0.3  # Proportion of warmup epochs.

    dataset: str = "imagenet"  # Dataset used

    def __post_init__(self):
        """Post initialization checks.

        Raises
        ------
        ValueError
            If the specified number of classes is not 10 or 100,
            the base model is not implemented,
            or if the model is topographic and a topographic parameter
            has not been specified.
        """
        if self.topographic and (
            self.lambd is None or self.norm is None or self.dimension is None
        ):
            raise ValueError(
                "If the model is set to be topographic, lambd, dimension and "
                + "norm must be provided."
            )
        if not self.topographic:
            self.lambd, self.dimension, self.norm = "None", "None", "None"
        try:
            getattr(models, self.model)
        except AttributeError as error:
            raise ValueError(
                f"The specified model is not implemented: {str(error)}"
            )


def main(config: ImageNetConfig) -> None:
    """Train a model on CIFAR with the given configuration.

    Parameters
    ----------
    config : CIFARConfig
        Pipeline configuration.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(config.crop_size),
            transforms.RandomHorizontalFlip(config.horizontal_flip),
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZATION),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(config.resize_size),
            transforms.CenterCrop(config.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZATION),
        ]
    )

    train_set = datasets.ImageNet(
        root=config.data, split="train", transform=train_transform
    )
    val_set = datasets.ImageNet(
        root=config.data, split="train", transform=test_transform
    )

    num_train = len(train_set)
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(num_train, generator=generator)
    train_idx, val_idx = indices[config.val_size :], indices[: config.val_size]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
    )

    test_set = datasets.ImageNet(
        root=config.data, split="val", transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    accelerator = Accelerator(log_with="tensorboard", logging_dir=config.log)
    project_name = (
        f"{datetime.now().strftime('%b%d_%H-%M-%S')}"
        + f"_{socket.gethostname()}_{uuid.uuid4()}"
    )
    accelerator.init_trackers(project_name, dataclasses.asdict(config))

    device = accelerator.device
    base_model = getattr(models, config.model)(num_classes=NUM_CLASSES)
    model = (
        TopographicModel(
            base_model,
            dimension=config.dimension,
            norm=config.norm,
            topographic_layer_names=models.topographic_layer_names(
                config.model
            ),
        ).to(device)
        if config.topographic
        else base_model.to(device)
    )
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.epochs * config.warmup_epochs_prop,
        max_epochs=config.epochs,
    )
    cross_entropy = nn.CrossEntropyLoss()
    topo_loss = TopographicLoss()

    if config.topographic:

        def criterion(output, target):
            ce = cross_entropy(output, target)
            topo = topo_loss(model.activations, model.inverse_distance)
            return MetricOutput(
                value=ce + config.lambd * topo.value,
                extras={
                    "loss-cross-entropy": ce.item(),
                    "loss-topographic": topo.value.item(),
                    **topo.extras,
                },
            )

    else:

        def criterion(output, target):
            return MetricOutput(value=cross_entropy(output, target))

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    checkpoints_path = Path(config.log) / "checkpoints"
    checkpoints_path.mkdir(exist_ok=True)

    start_epoch = 0
    existing_checkpoints = sorted(checkpoints_path.glob("*.model"))
    if existing_checkpoints:
        last_epoch = existing_checkpoints[-1].stem
        model.load_state_dict(
            torch.load(
                checkpoints_path / f"{last_epoch}.model",
                map_location=device,
            )
        )
        optimizer.load_state_dict(
            torch.load(
                checkpoints_path / f"{last_epoch}.optimizer",
                map_location=device,
            )
        )
        scheduler.load_state_dict(
            torch.load(
                checkpoints_path / f"{last_epoch}.scheduler",
                map_location=device,
            )
        )
        accelerator.print(f"Loaded checkpoint from epoch {last_epoch}")
        start_epoch = int(last_epoch) + 1

    for epoch in range(start_epoch, config.epochs):
        train_logs = train_accelerate(
            model, train_loader, optimizer, criterion, accelerator
        )
        accelerator.log(train_logs, step=epoch)
        accelerator.print(
            f"Train {epoch}: "
            + ", ".join([f"{k}: {v}" for k, v in train_logs.items()])
        )
        val_logs = evaluate_accelerate(
            model, val_loader, criterion, device, mode="val"
        )
        accelerator.log(val_logs, step=epoch)
        accelerator.print(
            f"Val {epoch}: "
            + ", ".join([f"{k}: {v}" for k, v in val_logs.items()])
        )
        scheduler.step()

        torch.save(model.state_dict(), checkpoints_path / f"{epoch:04d}.model")
        torch.save(
            optimizer.state_dict(), checkpoints_path / f"{epoch:04d}.optimizer"
        )
        torch.save(
            scheduler.state_dict(), checkpoints_path / f"{epoch:04d}.scheduler"
        )

    state_dict = sorted(checkpoints_path.glob("*.model"))[-1]
    model.load_state_dict(torch.load(state_dict, map_location=device))

    test_logs = evaluate_accelerate(
        model, test_loader, criterion, device, mode="test"
    )
    accelerator.log(test_logs)
    accelerator.end_training()


if __name__ == "__main__":
    # Some configuration parameters in CIFARConfig are not expected to be
    # changed. They are inside the CIFARConfig for code clarity
    # and to log them with the other hyperparameters.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Data directory."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--model", default="resnet18", type=str, help="Model to use."
    )
    parser.add_argument(
        "--topographic",
        action="store_true",
        help="If specified, use a topographic model.",
    )
    parser.add_argument(
        "--lambd", type=float, help="Weight of the topographic loss."
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="Dimension of the positions of the channels.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        help="Which norm between positions to use.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = ImageNetConfig(**vars(args))
    main(config)
