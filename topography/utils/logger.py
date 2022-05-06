"""Logging utilities.
"""
import logging

import topography


def get_logger(name: str, file: str) -> logging.Logger:
    """Create a logger with name `name` and that will output to
    `file`, with logging level `topography.LOG_LEVEL`.

    Parameters
    ----------
    name : str
        Logger name.
    file : str
        Logging file.

    Returns
    -------
    logging.Logger
        New logger.

    Raises
    ------
    ValueError
        If the global variable `topography.LOG_LEVEL` is not in
        ('debug', 'info', 'warning', 'error').
    """
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR
    }

    formatter = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    handler = logging.FileHandler(file)
    handler.setFormatter(logging.Formatter(formatter))

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.setLevel(levels[topography.LOG_LEVEL])
    except BaseException:
        raise ValueError(f'Invalid logging level "{topography.LOG_LEVEL}"')
    return logger
