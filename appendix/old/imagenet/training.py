"""Provides training and evaluation loops.
"""
import time
from typing import Callable, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from topography.base import Metric, MetricOutput
from topography.training.writer import Writer

PyTorchLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def accuracy(
    output: torch.Tensor, labels: torch.Tensor, topk: Tuple[int] = (1, 5)
) -> MetricOutput:
    """Computes the accuracy over the k top predictions for the
    specified values of k

    Parameters
    ----------
    output : torch.Tensor
        Logits outputs of the network.
    labels : torch.Tensor
        Target labels.

    Returns
    -------
    MetricOutput
        Accuracy on the given batch.
    """
    if 1 not in topk:
        raise ValueError("1 must be in topk in accuracy.")
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, predicted = output.topk(maxk, 1, True, True)
        predicted = predicted.t()
        correct = predicted.eq(labels.view(1, -1).expand_as(predicted))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f"acc/top{k}"] = correct_k.mul_(100.0 / batch_size).item()
        return MetricOutput(value=res["acc/top1"], extras=res)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: Union[Metric, PyTorchLoss],
    device: torch.device,
    writer: Writer,
    *,
    is_pytorch_loss: bool = False,
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
    criterion : Union[Metric, PyTorchLoss]
        Loss function.
    device : torch.device
        Device, either CPU or CUDA GPU.
    writer : Writer
        Writing utility.
    is_pytorch_loss: bool
        Has to be True if the given `criterion` is a loss from PyTorch.
        Else, it is considered to be a Metric and to return
        a MetricOutput with two fields: value and extras.
    """
    model.train()
    writer.next_epoch("train")
    end = time.time()

    if is_pytorch_loss:

        def metric_criterion(output, target):
            return MetricOutput(value=criterion(output, target))

    else:
        metric_criterion = criterion

    with tqdm(total=len(dataloader), desc=writer.desc()) as pbar:
        for batch_idx, (data, target) in enumerate(dataloader):
            # Measure data loading time
            writer["load-time"].update(time.time() - end)
            data, target = data.to(device), target.to(device)

            # Compute output
            output = model(data)
            loss = metric_criterion(output, target)

            # Compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.value.backward()
            optimizer.step()

            # Compute accuracy
            acc = accuracy(output, target)

            # Measure accuracy and record loss
            writer["loss"].update(loss.value.item(), data.size(0))
            writer["acc"].update(acc.value, data.size(0))
            for name, value in {**loss.extras, **acc.extras}.items():
                writer[f"extras/{name}"].update(value, data.size(0))

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
    criterion: Union[Metric, PyTorchLoss],
    device: torch.device,
    writer: Writer,
    *,
    mode: str = "test",
    is_pytorch_loss: bool = False,
) -> None:
    """Testing or validation loop.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    dataloader : DataLoader
        Dataloader.
    criterion : Union[Metric, PyTorchLoss]
        Loss function.
    device : torch.device
        Device, either CPU or CUDA GPU.
    writer : Writer
        Writing utility.
    mode : str
        Evaluation mode ('test' or 'val'), by default 'test'.
    is_pytorch_loss: bool
        Has to be True if the given `criterion` is a loss from PyTorch.
        Else, it is considered to be a Metric and to return
        a MetricOutput with two fields: value and extras.
    """
    model.eval()
    writer.next_epoch(mode)

    if is_pytorch_loss:

        def metric_criterion(output, target):
            return MetricOutput(value=criterion(output, target))

    else:
        metric_criterion = criterion

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = metric_criterion(output, target)
            acc = accuracy(output, target)
            writer["loss"].update(loss.value.item(), data.size(0))
            writer["acc"].update(acc.value, data.size(0))
            for name, value in {**loss.extras, **acc.extras}.items():
                writer[f"extras/{name}"].update(value, data.size(0))
            writer.log(batch_idx)
        print(writer.summary())
