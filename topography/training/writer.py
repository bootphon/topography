from collections import namedtuple
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from topography.utils import AverageMeter, get_logger

Metric = namedtuple('Metric', ['meter', 'compute'])


class Writer:
    def __init__(self,
                 mode: str,
                 root: str,
                 epoch: int,
                 metrics: Optional[List[Tuple[str, str, Callable]]] = None,
                 ) -> None:
        self.root = Path(root)
        self.epoch = epoch

        self.logs = self.root.joinpath('logs')
        self.models = self.root.joinpath('models')

        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        self.models.mkdir(exist_ok=True)

        if epoch is not None:
            self.logger = get_logger(
                mode+f', epoch {epoch}',
                self.logs.joinpath(f'{mode.lower()}-{epoch:03d}.log'))
        else:
            self.logger = get_logger(
                mode, self.logs.joinpath(f'{mode.lower()}.log'))
        self.summary_logger = get_logger(
            'Summary', self.root.joinpath('summary.log'))

        self._batch_time = AverageMeter('time', ':.3f')
        self._data_time = AverageMeter('data', ':.3f')
        self._losses = AverageMeter('loss', ':.3e')
        self._metrics = []
        if metrics is not None:
            self._metrics = [Metric(AverageMeter(m[0], m[1]), m[2])
                             for m in metrics]

    def save(self, model, optimizer):
        torch.save(model.state_dict(), self.models.joinpath(
                   f"{self.epoch:03d}.model"))
        torch.save(optimizer.state_dict(), self.models.joinpath(
                   f"{self.epoch:03d}.optim"))

    def end(self, t):
        self.logger.info(f'Finished epoch in {t}s.')
        metrics = {
            'batch_time': self._batch_time,
            'losses': self._losses,
            'data_time': self._data_time
        }
        for m in self._metrics:
            metrics[m.meter.name] = m.meter
        metrics['epoch_time'] = t
        return metrics

    def update_data_time(self, t):
        self._data_time.update(t)

    def update_batch_time(self, t):
        self._batch_time.update(t)

    def update_losses(self, loss, batch_size):
        self._losses.update(loss, batch_size)

    def update_metrics(self, output, target, batch_size):
        for m in self._metrics:
            m.meter.update(m.compute(output, target), batch_size)

    def log(self, batch_idx: int):
        message = f'Batch {batch_idx}, {self._data_time}, '\
            f'{self._batch_time}, '\
            f'{self._losses}'
        for m in self._metrics:
            message += f', {m.meter}'
        self.logger.debug(message)

    def current(self, head: str = ''):
        meters = [self._losses]
        for m in self._metrics:
            meters.append(m.meter)
        return head+', '.join([str(meter) for meter in meters])

    def summary(self, head: str = ''):
        meters = [self._losses]
        for m in self._metrics:
            meters.append(m.meter)
        out = head+', '.join([meter.summary() for meter in meters])
        self.summary_logger.info(out)
        return out
