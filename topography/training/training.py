import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from topography.training.writer import Writer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def accuracy(output, labels):
    _, predicted = torch.max(output.data, 1)
    return float((predicted == labels).sum()) / float(output.size(0))


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    device: torch.device,
    save_dir: str,
    epoch: int,
) -> None:
    model.train()
    writer = Writer('Train', save_dir, epoch,
                    metrics=[('acc', ':.2f', accuracy)])
    end = time.time()

    with tqdm(total=len(dataloader), desc=f'Train, epoch {epoch}') as pbar:
        for batch_idx, (data, target) in enumerate(dataloader):
            # Measure data loading time
            writer.update_data_time(time.time() - end)
            data, target = data.to(device), target.to(device)

            # Compute output
            output = model(data)
            loss = criterion(output, target)

            # Compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Measure accuracy and record loss
            writer.update_losses(loss.item(), data.size(0))
            writer.update_metrics(output, target, data.size(0))

            # Measure elapsed time
            writer.update_batch_time(time.time() - end)
            end = time.time()

            pbar.set_postfix_str(writer.current())
            pbar.update()
            writer.log(batch_idx)
        writer.summary(f'Train, epoch {epoch}: ')
    writer.save(model, optimizer)
    return writer.end(time.time() - end)


def test(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
    save_dir: str,
    mode: str,
    epoch: Optional[int] = None
) -> None:
    model.eval()
    writer = Writer(mode, save_dir, epoch,
                    metrics=[('acc', ':.2f', accuracy)])
    end = time.time()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()

            writer.update_losses(loss, data.size(0))
            writer.update_metrics(output, target, data.size(0))
            writer.update_batch_time(time.time() - end)
            end = time.time()
            writer.log(batch_idx)

        if epoch is not None:
            head = f'{mode}, epoch {epoch}: '
        else:
            head = f'{mode}: '
        print(writer.summary(head))
    return writer.end(time.time() - end)
