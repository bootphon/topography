"""Base types and dataclasses."""
import dataclasses
from collections.abc import Mapping
from typing import Callable, Dict, OrderedDict, Union

import torch


class MappingDataclass(Mapping):
    def __len__(self):
        return len(dataclasses.asdict(self))

    def __getitem__(self, item):
        return dataclasses.asdict(self)[item]

    def __iter__(self):
        return iter(dataclasses.asdict(self))


@dataclasses.dataclass
class MetricOutput(MappingDataclass):
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
