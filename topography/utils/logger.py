"""Logging utilities.
"""
import logging

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
