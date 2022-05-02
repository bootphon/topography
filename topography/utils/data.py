from typing import List, Optional, Sequence, TypeVar

from torch import Generator, default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int],
                 transforms) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transforms = transforms

    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        return self.transforms(data), label

    def __len__(self):
        return len(self.indices)


def random_split(dataset: Dataset[T], lengths: Sequence[int],
                 transforms: Compose,
                 generator: Optional[Generator] = default_generator) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [Subset(dataset, indices[offset - length: offset], transform) for
            offset, length, transform in zip(
                _accumulate(lengths), lengths, transforms)]
