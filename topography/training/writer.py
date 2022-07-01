"""Writing utility. Handle logging, writing to TensorBoard and saving
checkpoints.
"""
import inspect
import json
import shutil
import socket
import subprocess
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

import topography
from topography.utils import AverageMeter, get_logger

_COMMANDS_STDOUT = (
    ("conda env export", "env_with_builds.yml"),
    ("conda env export --no-builds", "env_no_builds.yml"),
)


def _copy_git_directory(path: Path) -> None:
    """Copy the git folder associated to this project to `path`.
    Used for reproducibility

    Parameters
    ----------
    path : Path
        Output path.
    """
    git_path = Path(inspect.getfile(topography)).parent.parent.joinpath(".git")
    if git_path.exists():
        shutil.copytree(git_path, path)


def _exec_command(cmd: str, path: Optional[Path] = None) -> None:
    """Run a given command and writes its output to `path`.

    Parameters
    ----------
    cmd : str
        Command to run
    path : Optional[Path], optional
        Path where the output is written, by default None
    """
    try:
        out = subprocess.check_output(cmd.split())
    except subprocess.CalledProcessError as error:  # pragma: no cover
        print(f"Command {cmd} failed: {str(error)}")
        return
    with open(path, "w", encoding="utf-8") as file:
        file.write(out.decode("utf-8"))


class Writer:
    """Writer handling logging, TensorBoard and checkpoints"""

    def __init__(self, log_dir: str, fmt: str = ":.3f") -> None:
        """Create the writer.

        Parameters
        ----------
        log_dir : str
            Logging directory. It will be structured in the following way:

            log_dir/
            |--checkpoints/
            |--environment/
            |--tensorboard/
            |
            |--summary.log
            |--train.log
            |--val.log
            |--test.log
        fmt : str, optional
            String formatter used in logging, by default ':.3f'.
        """
        self._start_time = datetime.now()
        self.root = (
            Path(log_dir)
            .joinpath(
                f"{self._start_time.strftime('%b%d_%H-%M-%S')}"
                f"_{socket.gethostname()}_{uuid.uuid4()}"
            )
            .resolve()
        )
        self.fmt = fmt
        self.tensorboard = SummaryWriter(self.root.joinpath("tensorboard"))
        self._checkpoints = self.root.joinpath("checkpoints")
        self._environment = self.root.joinpath("environment")

        self.root.mkdir(parents=True, exist_ok=True)
        self._checkpoints.mkdir(exist_ok=True)
        self._environment.mkdir(exist_ok=True)

        self._summary_logger = get_logger(
            "summary", self.root.joinpath("summary.log")
        )
        self._mode = None
        self._meters = {}
        self._loggers = {}
        self._epochs = {}
        self._to_remove = "extras"

        self._summary_logger.info("Start on %s.", self._start_time)
        for cmd, path in _COMMANDS_STDOUT:
            _exec_command(cmd, self._environment.joinpath(path))
        _copy_git_directory(self._environment.joinpath("git_directory"))

    def __getitem__(self, metric: str) -> AverageMeter:
        """Return the meter associated for the given `metric`
        for the current epoch and the current mode.

        Parameters
        ----------
        metric : str
            Metric to get. Has to have been given by `set`
            for this current epoch and mode, else an empty AverageMeter
            will be created.

        Returns
        -------
        AverageMeter
            Associated meter.
        """
        current_meters = self._meters[self._mode][self._epochs[self._mode]]
        if metric not in current_meters:
            current_meters[metric] = AverageMeter(metric, self.fmt)
        return current_meters[metric]

    def next_epoch(self, mode: str) -> None:
        """Start a new epoch with the given `mode`, will track
        the given `metrics`. If the `mode` has not been seen yet,
        it will be the first epoch.

        Parameters
        ----------
        mode : str
            Current mode, for example "train", "val", or "test".
        """
        self._mode = mode
        if mode not in self._meters:
            self._meters[mode] = OrderedDict()
            self._loggers[mode] = get_logger(
                mode, self.root.joinpath(f"{mode}.log")
            )
            self._epochs[mode] = 1
        else:
            self._epochs[mode] += 1
        self._meters[mode][self._epochs[mode]] = OrderedDict()

    def desc(self) -> str:
        """String indicating the current mode and epoch.
        Used in tqdm progress bar.

        Returns
        -------
        str
            Description of the Writer state.
        """
        return f"{self._mode}, epoch {self._epochs[self._mode]}"

    def save(
        self,
        mode: str,
        metric: str,
        maximize: bool = True,
        clean: bool = True,
        **kwargs,
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
        clean : bool, optional
            Whether to clean the checkpoints directory and delete previous
            checkpoints if the best score is attained. By default True.
        **kwargs:
            Torch objects that we wish to save. Must implement the
            `state_dict` method (ie models, optimizers, schedulers).
        """
        scores_per_epoch = [m[metric].avg for m in self._meters[mode].values()]
        last_score = scores_per_epoch[-1]
        if maximize:
            cond = all(last_score >= score for score in scores_per_epoch)
        else:
            cond = all(last_score <= score for score in scores_per_epoch)
        if cond:
            if clean:
                for file in self._checkpoints.glob("*"):
                    file.unlink()
            for k, module in kwargs.items():
                torch.save(
                    module.state_dict(),
                    self._checkpoints.joinpath(f"{self._epochs[mode]:04d}.{k}"),
                )

    def log(self, batch_idx: int) -> None:
        """Log to a text file intermediate results for batch number
        `batch_idx`.

        Parameters
        ----------
        batch_idx : int
            Batch number at the current epoch.
        """
        message = f"epoch {self._epochs[self._mode]}, batch {batch_idx}, "
        meters = self._meters[self._mode][self._epochs[self._mode]].values()
        message += ", ".join([str(meter) for meter in meters])
        self._loggers[self._mode].debug(message)

    def log_config(self, kwargs) -> None:
        """Log the given hyperparameters to a json file."""
        kwargs.pop("log", None)
        kwargs["log"] = str(self.root)
        with open(
            self._environment.joinpath("config.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(kwargs, file, indent=2)

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
        return ", ".join(
            [
                str(meter)
                for meter in meters
                if not meter.name.startswith(self._to_remove)
            ]
        )

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
            self.tensorboard.add_scalar(
                f"{self._mode}/{meter.name}",
                meter.avg,
                self._epochs[self._mode],
            )
        out = self.desc() + ", "
        out += ", ".join(
            [
                meter.summary()
                for meter in meters
                if not meter.name.startswith(self._to_remove)
            ]
        )
        self._summary_logger.info(out)
        return out

    def close(self) -> None:
        """Close the TensorBoard writer."""
        end = datetime.now()
        self._summary_logger.info("Ended on %s.", end)
        self._summary_logger.info("Lasted for %s.", end - self._start_time)
        self.tensorboard.close()
