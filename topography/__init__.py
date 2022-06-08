"""Inducing topographic organization in ConvNets."""
from topography.base import MetricOutput
from topography.core import TopographicLoss, TopographicModel, topographic_loss

LOG_LEVEL: str = "debug"

__all__ = [
    "TopographicLoss",
    "TopographicModel",
    "topographic_loss",
    "MetricOutput",
]
