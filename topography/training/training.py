"""Provides training and evaluation loops.
"""
import time
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from topography.training.writer import Writer


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute batch accuracy.

    Parameters
    ----------
    output : torch.Tensor
        Raw outputs of the network.
    labels : torch.Tensor
        Target labels.

    Returns
    -------
    float
        Accuracy on the given batch.
    """
    _, predicted = torch.max(output.data, 1)
    return float((predicted == labels).sum()) / float(output.size(0))


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    device: torch.device,
    writer: Writer,
) -> None:
    """Training loop.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    dataloader : DataLoader
        Dataloader.
    optimizer : Optimizer
        Optimizer.
    criterion : Callable
        Loss function.
    device : torch.device
        Device, either CPU or CUDA GPU.
    writer : Writer
        Writing utility.
    """
    model.train()
    writer.set("train", ["loss", "acc", "batch-time", "load-time"])
    end = time.time()

    with tqdm(total=len(dataloader), desc=writer.desc()) as pbar:
        for batch_idx, (data, target) in enumerate(dataloader):
            # Measure data loading time
            writer["load-time"].update(time.time() - end)
            data, target = data.to(device), target.to(device)

            # Compute output
            output = model(data)
            loss = criterion(output, target)

            # Compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Measure accuracy and record loss
            writer["loss"].update(loss.item(), data.size(0))
            writer["acc"].update(accuracy(output, target), data.size(0))

            # Measure elapsed time
            writer["batch-time"].update(time.time() - end)
            end = time.time()

            pbar.set_postfix_str(writer.postfix())
            pbar.update()
            writer.log(batch_idx)
        writer.summary()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
    writer: Writer,
    mode: str = "test",
) -> None:
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
    writer : Writer
        Writing utility.
    mode : str
        Evaluation mode ('test' or 'val'), by default 'test'.
    """
    model.eval()
    writer.set(mode, ["loss", "acc"])

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            acc = accuracy(output, target)
            writer["loss"].update(loss, data.size(0))
            writer["acc"].update(acc, data.size(0))
            writer.log(batch_idx)
        print(writer.summary())
