"""Base types and dataclasses."""
import dataclasses
from typing import Callable, Dict, OrderedDict, Union

import torch


@dataclasses.dataclass
class MetricOutput:
    """Output of a Metric. Used to store extra information other
    than the main value.
    """

    value: Union[float, torch.Tensor]
    extras: Dict[str, float] = dataclasses.field(default_factory=dict)


TensorDict = OrderedDict[str, torch.Tensor]
Metric = Union[
    Callable[[torch.Tensor, torch.Tensor], MetricOutput],
    Callable[[TensorDict, TensorDict], MetricOutput],
]
