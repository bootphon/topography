from collections import defaultdict
from pathlib import Path
from typing import List, NamedTuple, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
from torchaudio import datasets, transforms
from tqdm.auto import tqdm

from topography.utils.externals.meter import AverageMeter

_LABELS = {
    "backward": 0,
    "bed": 1,
    "bird": 2,
    "cat": 3,
    "dog": 4,
    "down": 5,
    "eight": 6,
    "five": 7,
    "follow": 8,
    "forward": 9,
    "four": 10,
    "go": 11,
    "happy": 12,
    "house": 13,
    "learn": 14,
    "left": 15,
    "marvin": 16,
    "nine": 17,
    "no": 18,
    "off": 19,
    "on": 20,
    "one": 21,
    "right": 22,
    "seven": 23,
    "sheila": 24,
    "six": 25,
    "stop": 26,
    "three": 27,
    "tree": 28,
    "two": 29,
    "up": 30,
    "visual": 31,
    "wow": 32,
    "yes": 33,
    "zero": 34,
}


class Metadata(NamedTuple):
    """Metadata container."""

    idx: str
    sample_rate: int
    label: str
    speaker_id: str
    utterance_number: int

    @classmethod
    def from_csv(cls, *row: List[str]):
        """Create a metadata entry from a CSV row."""
        return cls(int(row[0]), int(row[1]), row[2], row[3], row[4])

    def __str__(self) -> str:
        """String representation, used for CSV export."""
        return ",".join([str(field) for field in self])


def _build_dataset(
    src: str,
    dest: str,
    sample_rate: int,
    transform_fn: nn.Module = transforms.MelSpectrogram,
    **kwargs,
) -> None:
    transform = transform_fn(sample_rate=sample_rate, **kwargs)
    all_metadata = defaultdict(list)
    mean, std = AverageMeter("mean"), AverageMeter("std")
    for subset in ["training", "validation", "testing"]:
        out = Path(dest).joinpath(subset)
        out.mkdir(parents=True, exist_ok=True)
        dataset = datasets.SPEECHCOMMANDS(
            root=src, download=True, subset=subset
        )
        for idx, sample in enumerate(tqdm(dataset, desc=subset, leave=False)):
            n_digits = len(str(len(dataset)))
            waveform, *other = sample
            metadata = Metadata(idx, *other)
            all_metadata[subset].append(str(metadata))
            if metadata.sample_rate != sample_rate:
                raise ValueError("Sample rate inconsistency.")
            if waveform.shape[1] < sample_rate:
                temp = torch.zeros(1, sample_rate)
                temp[:, : waveform.shape[1]] = waveform
                waveform = temp
            elif waveform.shape[1] > sample_rate:
                raise ValueError(
                    f"Sample {idx} in {subset} lasts for more than 1 second."
                )
            feats = transform(waveform)
            torch.save(feats, out.joinpath(f"{idx:0{n_digits}d}.pt"))
            if subset == "training":
                mean.update(feats.mean())
                std.update(feats.std())
        if subset == "training":
            torch.save(
                {"mean": mean.avg, "std": std.avg},
                Path(dest).joinpath("training_stats.pt"),
            )
    for subset, metadata in all_metadata.items():
        with open(
            dest.joinpath(f"{subset}.csv"), "w", encoding="utf-8"
        ) as file:
            file.write("\n".join(metadata))


class SpeechCommands(Dataset):
    _sample_rate = 16_000
    num_classes = 35

    def __init__(
        self,
        root: str,
        subset: str,
        build: bool = False,
        transform_fn: nn.Module = transforms.MelSpectrogram,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = Path(root).joinpath("speechcommands").resolve()
        self._subset = subset
        if build:
            _build_dataset(
                root, self.root, self._sample_rate, transform_fn, **kwargs
            )
        stats = torch.load(self.root.joinpath("training_stats.pt"))
        self._mean, self._std = stats["mean"], stats["std"]
        self.metadata = {}
        with open(self.root.joinpath(f"{subset}.csv"), "r") as file:
            for line in file.readlines():
                idx, *other = line.split(",")
                self.metadata[int(idx)] = Metadata.from_csv(idx, *other)
        self._path = self.root.joinpath(self._subset)
        self._n_digits = len(str(len(self.metadata)))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        metadata = self.metadata[index]
        path = self._path.joinpath(f"{index:0{self._n_digits}d}.pt")
        normalized = (torch.load(path) - self._mean) / self._std
        return normalized, _LABELS[metadata.label]

    def __len__(self) -> int:
        return len(self.metadata)
