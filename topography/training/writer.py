"""Writing utility. Handle logging, writing to TensorBoard and saving
checkpoints.
"""
import json
import socket
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch.utils.tensorboard import SummaryWriter

from topography.utils import AverageMeter, get_logger


class Writer:
    def __init__(self, log_dir: str, fmt: str = ':.3f') -> None:
        """Writer handling logging, TensorBoard and checkpoints.

        Parameters
        ----------
        log_dir : str
            Logging directory. It will be structured in the following way:

            log_dir/
            |--checkpoints/
            |
            |--tensorboard/
            |
            |--summary.log
            |--train.log
            |--val.log
            |--test.log

        fmt : str, optional
            String formatter used in logging, by default ':.3f'.
        """
        # TODO: redisign the writer: a bit convoluted that way, have to use
        # TODO: more TensorBoard or even Wandb.
        time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.root = Path(log_dir).joinpath(f'{time}_{socket.gethostname()}')
        self.fmt = fmt
        self.tb = SummaryWriter(self.root.joinpath('tensorboard'))
        self._checkpoints = self.root.joinpath('checkpoints')

        self.root.mkdir(parents=True, exist_ok=True)
        self._checkpoints.mkdir(exist_ok=True)
        self._summary_logger = get_logger(
            'summary', self.root.joinpath('summary.log'))
        self._meters = {}
        self._loggers = {}
        self._epochs = {}

    def __getitem__(self, metric: str) -> AverageMeter:
        """Return the meter associated for the given `metric`
        for the current epoch and the current mode.

        Parameters
        ----------
        metric : str
            Metric to get. Has to have been given by `set`
            for this current epoch and mode.

        Returns
        -------
        AverageMeter
            Associated meter.
        """
        return self._meters[self._mode][self._epochs[self._mode]][metric]

    def set(self, mode: str, metrics: List[str]) -> None:
        """Start a new epoch with the given `mode`, will track
        the given `metrics`. If the `mode` has not been seen yet,
        it will be the first epoch.

        Parameters
        ----------
        mode : str
            Current mode, for example "train", "val", or "test".
        metrics : List[str]
            List of metrics to follow for this mode and the current epoch.
        """
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
        """String indicating the current mode and epoch.
        Used in tqdm progress bar?

        Returns
        -------
        str
            Description of the Writer state.
        """
        return f'{self._mode}, epoch {self._epochs[self._mode]}'

    def save(self, mode: str, metric: str, maximize: bool = True, **kwargs
             ) -> None:
        """Save checkpoints if for the given `mode`, the score for `metric`
        is the best at the current epoch.

        Parameters
        ----------
        mode : str
            Mode in which the scores will be compared across epochs.
            Most likely will be "val" and not "train".
        metric : str
            Metric that will decide if we save the checkpoints or not.
            Most likely will be "acc".
        maximize : bool, optional
            Whether we look to maximize or minimize this metric, by default
            True. Most likely will be True if `metric` is "acc", and
            False if `metric` is "loss".
        **kwargs:
            Torch objects that we which to save. Must implement the
            `state_dict` method (ie models, optimizers, schedulers).
        """
        scores_per_epoch = [m[metric].avg for m in self._meters[mode].values()]
        last_score = scores_per_epoch[-1]
        if maximize:
            cond = all([last_score >= score for score in scores_per_epoch])
        else:
            cond = all([last_score <= score for score in scores_per_epoch])
        if cond:
            for k, v in kwargs.items():
                torch.save(v.state_dict(), self._checkpoints.joinpath(
                    f'{self._epochs[mode]:04d}.{k}'))

    def log(self, batch_idx: int) -> None:
        """Log to a text file intermediate results for batch number
        `batch_idx`.

        Parameters
        ----------
        batch_idx : int
            Batch number at the current epoch.
        """
        message = f'epoch {self._epochs[self._mode]}, batch {batch_idx}, '
        meters = self._meters[self._mode][self._epochs[self._mode]].values()
        message += ', '.join([str(meter) for meter in meters])
        self._loggers[self._mode].debug(message)

    def log_hparams(self, **kwargs) -> None:
        """Log the given hyperparameters to a json file."""
        with open(self.root.joinpath('hparams.json'), 'w') as f:
            json.dump(kwargs, f, indent=2)

    def postfix(self) -> str:
        """Postfix showing the state of each tracked metric for the
        current mode and epoch. Plain scores are those of the last batch while
        scores in parenthesis are averages on the current epoch.

        Returns
        -------
        str
            Postfix string. Used in tqdm progress bar.
        """
        meters = self._meters[self._mode][self._epochs[self._mode]].values()
        return ', '.join([str(meter) for meter in meters])

    def summary(self) -> str:
        """Summary of all tracked metrics on the current epoch and mode.
        Also log epoch results to TensorBoard.

        Returns
        -------
        str
            Summary string.
        """
        meters = self._meters[self._mode][self._epochs[self._mode]].values()
        for meter in meters:
            self.tb.add_scalar(f'{self._mode}/{meter.name}',
                               meter.avg, self._epochs[self._mode])
        out = self.desc() + ', '
        out += ', '.join([meter.summary() for meter in meters])
        self._summary_logger.info(out)
        return out

    def close(self) -> None:
        """Close the TensorBoard writer.
        """
        self.tb.close()
