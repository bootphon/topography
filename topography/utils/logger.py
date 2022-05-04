"""Logging utilities.
"""
import logging

LOG_LEVEL: str = 'debug'


def get_logger(name: str, file: str) -> logging.Logger:
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
        logger.setLevel(levels[LOG_LEVEL])
    except BaseException:
        raise ValueError(f'Invalid logging level "{LOG_LEVEL}"')
    return logger
