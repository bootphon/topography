import socket
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from topography.utils import AverageMeter, get_logger
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, log_dir: str, fmt: str = ':.3f') -> None:
        time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.root = Path(log_dir).joinpath(f'{time}_{socket.gethostname()}')
        self.fmt = fmt
        self._tensorboard_writer = SummaryWriter(
            self.root.joinpath('tensorboard'))
        self._models = self.root.joinpath('models')

        self.root.mkdir(parents=True, exist_ok=True)
        self._models.mkdir(exist_ok=True)
        self._summary_logger = get_logger(
            'summary', self.root.joinpath('summary.log'))

        self._meters = {}
        self._loggers = {}
        self._epochs = {}

    def __getitem__(self, metric: str) -> AverageMeter:
        return self._meters[self._mode][self._epochs[self._mode]][metric]

    def set(self, mode: str, metrics: List[str]) -> None:
        self._mode = mode
        if mode not in self._meters:
            self._meters[mode] = OrderedDict()
            self._loggers[mode] = get_logger(
                mode, self.root.joinpath(f'{mode}.log'))
            self._epochs[mode] = 1
        else:
            self._epochs[mode] += 1
        self._meters[mode][self._epochs[mode]] = OrderedDict(
            [(m, AverageMeter(m, self.fmt)) for m in metrics])

    def desc(self) -> str:
        return f'{self._mode}, epoch {self._epochs[self._mode]}'

    def save(self, model: nn.Module, optimizer: Optimizer, mode: str,
             metric: str, maximize: bool = True) -> None:
        scores_per_epoch = [m[metric].avg for m in self._meters[mode].values()]
        last_score = scores_per_epoch[-1]
        if maximize:
            cond = all([last_score >= score for score in scores_per_epoch])
        else:
            cond = all([last_score <= score for score in scores_per_epoch])
        if cond:
            torch.save(model.state_dict(), self._models.joinpath(
                f'{self._epochs[mode]:04d}.model'))
            torch.save(optimizer.state_dict(), self._models.joinpath(
                f'{self._epochs[mode]:04d}.optim'))

    def log(self, batch_idx: int) -> None:
        message = f'epoch {self._epochs[self._mode]}, batch {batch_idx}, '
        meters = self._meters[self._mode][self._epochs[self._mode]].values()
        message += ', '.join([str(meter) for meter in meters])
        self._loggers[self._mode].debug(message)

    def postfix(self) -> str:
        meters = self._meters[self._mode][self._epochs[self._mode]].values()
        return ', '.join([str(meter) for meter in meters])

    def summary(self) -> str:
        meters = self._meters[self._mode][self._epochs[self._mode]].values()
        for meter in meters:
            self._tensorboard_writer.add_scalar(
                f'{self._mode}/{meter.name}',
                meter.avg,
                self._epochs[self._mode]
            )
        out = self.desc() + ', '
        out += ', '.join([meter.summary() for meter in meters])
        self._summary_logger.info(out)
        return out

    def close(self) -> None:
        self._tensorboard_writer.close()
