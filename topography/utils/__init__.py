"""Utility functions.
This learning rate scheduler is used in [tricks]_.

References
----------
.. [tricks] https://arxiv.org/abs/1812.01187
"""
from topography.utils.externals.meter import AverageMeter
from topography.utils.externals.scheduler import LinearWarmupCosineAnnealingLR
from topography.utils.logger import get_logger

__all__ = ["AverageMeter", "LinearWarmupCosineAnnealingLR", "get_logger"]
