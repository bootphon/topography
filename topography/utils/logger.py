"""Logging utilities.
"""
import logging
from pathlib import Path
from typing import Union

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

import topography


def get_logger(name: str, file: str, level: str = None) -> logging.Logger:
    """Create a logger with name `name` and that will output to
    `file`, with logging level `topography.LOG_LEVEL`.

    Parameters
    ----------
    name : str
        Logger name.
    file : str
        Logging file.
    level : str, optional
        Logging level. If not provided, uses the global variable
        `topography.LOG_LEVEL` instead.

    Returns
    -------
    logging.Logger
        New logger.

    Raises
    ------
    ValueError
        If the logging level is not in ('debug', 'info', 'warning', 'error').
    """
    if level is None:
        level = topography.LOG_LEVEL
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    formatter = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    handler = logging.FileHandler(file)
    handler.setFormatter(logging.Formatter(formatter))

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.setLevel(levels[level])
    except BaseException as invalid_level:
        raise ValueError(f'Invalid logging level "{level}"') from invalid_level
    return logger


def tensorboard_to_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """Convert a TensorBoard file to a pd.DataFrame.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the TensorBoard file.

    Returns
    -------
    pd.DataFrame
        New DataFrame.
    """
    path = str(path)
    event_acc = event_accumulator.EventAccumulator(path)
    dataframe = pd.DataFrame({"metric": [], "value": [], "step": []})
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]
    for tag in tags:
        event_list = event_acc.Scalars(tag)
        values = [event.value for event in event_list]
        step = [event.step for event in event_list]
        rows = {"metric": [tag] * len(step), "value": values, "step": step}
        dataframe = pd.concat(
            [dataframe, pd.DataFrame(rows)], ignore_index=True
        )
    return dataframe
