"""Script to train a model, topographic or not, on SpeechCommands."""
import argparse
import dataclasses
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from topography import MetricOutput, TopographicLoss, TopographicModel, models
from topography.training import Writer, evaluate, train
from topography.utils import LinearWarmupCosineAnnealingLR
from topography.utils.data import SpeechCommands

NUM_CLASSES, IN_CHANNELS = 35, 1


@dataclasses.dataclass(frozen=True)
class SpeechCommandsConfig:
    """Dataclass to store the whole configuration used."""

    log: str  # Output directory.
    data: str  # Data directory.
    seed: int  # Random seed.

    model: str  # Model to use.
    topographic: bool  # Whether to train a topographic model or not.
    lambd: Optional[float] = None  # Weight of the topographic loss.
    dimension: Optional[int] = None  # Dimension of the positions.
    norm: Optional[str] = None  # Which norm between positions to use.

    epochs: int = 12  # Number of training epochs.
    batch_size: int = 256  # Batch size.
    lr: float = 0.01  # Base learning rate.
    weight_decay: float = 0.01  # Weight decay.
    momentum: float = 0.9  # SGD momentum.
    optimizer: str = "sgd"  # Optimizer.
    scheduler: str = "LinearWarmupCosineAnnealingLR"  # LR scheduler.
    warmup_epochs_prop: float = 0.3  # Proportion of warmup epochs.

    def __post_init__(self):
        """Post initialization checks.

        Raises
        ------
        ValueError
            If the specified number of classes is not 10 or 100,
            the base model is not implemented,
            or if the model is topographic and lambd has not been specified.
        """
        if self.topographic and (
            self.lambd is None or self.norm is None or self.dimension is None
        ):
            raise ValueError(
                "If the model is set to be topographic, lambd, dimension and "
                + "norm must be provided."
            )
        try:
            getattr(models, self.model)
        except AttributeError as error:
            raise ValueError(
                f"The specified model is not implemented: {str(error)}"
            )


def main(config: SpeechCommandsConfig) -> None:
    """Train a model on CIFAR with the given configuration.

    Parameters
    ----------
    config : SpeechCommandsConfig
        Pipeline configuration.
    """
    writer = Writer(config.log)

    train_set = SpeechCommands(config.data, subset="training")
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_set = SpeechCommands(config.data, subset="validation")
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_set = SpeechCommands(config.data, subset="testing")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = getattr(models, config.model)(
        num_classes=NUM_CLASSES, in_channels=IN_CHANNELS
    )
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

    writer.log_config(dataclasses.asdict(config))
    for _ in range(config.epochs):
        train(model, train_loader, optimizer, criterion, device, writer)
        evaluate(model, val_loader, criterion, device, writer, mode="val")
        scheduler.step()
        writer.save(
            "val", "acc", model=model, optimizer=optimizer, scheduler=scheduler
        )

    state_dict = sorted((writer.root / "checkpoints").glob("*.model"))[-1]
    model.load_state_dict(torch.load(state_dict, map_location=device))

    evaluate(model, test_loader, criterion, device, writer, mode="test")
    writer.close()


if __name__ == "__main__":
    # Some configuration parameters in SpeechCommandsConfig are not expected
    # to be changed. They are inside the SpeechCommandsConfig for code clarity
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

    config = SpeechCommandsConfig(**vars(args))
    main(config)
