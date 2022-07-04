"""Wraps the SPEECHCOMMANDS dataset from torchaudio
to use pre-processed features.
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
from torchaudio import datasets
from torchaudio.datasets.speechcommands import FOLDER_IN_ARCHIVE
from tqdm.auto import tqdm

from topography.utils import AverageMeter
from topography.utils.data.common import default_audio_transform

_LABELS: Dict[str, int] = {
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


class SpeechCommandsMetadata(NamedTuple):
    """Metadata container."""

    idx: int
    sample_rate: int
    label: str
    speaker_id: str
    utterance_number: int

    @classmethod
    def from_csv(cls, *row: str):
        """Create a metadata entry from a CSV row."""
        return cls(int(row[0]), int(row[1]), row[2], row[3], int(row[4]))

    def __str__(self) -> str:
        """String representation, used for CSV export."""
        return ",".join(
            [str(field) for field in self]  # pylint: disable=not-an-iterable
        )


def _build_dataset(
    src: str,
    dest: str,
    sample_rate: int,
    download: bool,
    transform: nn.Module,
) -> None:  # pragma: no cover
    """Build the processed Speech Commands dataset.
    Samples are padded with zeros for them to last for exactly one second.

    Parameters
    ----------
    src : str
        Source directory where the raw dataset from torchaudio
        is to be found or downloaded.
    dest : str
        Destination directory.
    sample_rate : int
        Sample rate of Speech Commands.
    download : bool
        Whether to download the dataset if it is not found at root path.
    transform : nn.Module
        Transformation to apply to the audio.

    Raises
    ------
    ValueError
        If a sample does not have the same sample rate as the one
        specified or if a sample in the dataset lasts for more
        than 1 second.
    """
    all_metadata = defaultdict(list)
    mean, std = AverageMeter("mean"), AverageMeter("std")

    for subset in ["training", "validation", "testing"]:
        out = Path(dest).joinpath(subset)
        out.mkdir(parents=True, exist_ok=True)
        dataset = datasets.SPEECHCOMMANDS(
            root=src, download=download, subset=subset
        )

        for idx, sample in enumerate(tqdm(dataset, desc=subset, leave=False)):
            n_digits = len(str(len(dataset)))
            waveform, *other = sample
            metadata = SpeechCommandsMetadata(idx, *other)
            all_metadata[subset].append(str(metadata))

            # Check the sample and padding if necessary
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

            # Transformation and update the statistics
            feats = transform(waveform)
            torch.save(feats, out.joinpath(f"{idx:0{n_digits}d}.pt"))
            if subset == "training":
                mean.update(feats.mean())
                std.update(feats.std())

        # Save the statistics on the training set
        if subset == "training":
            torch.save(
                {"mean": mean.avg, "std": std.avg},
                Path(dest).joinpath("training_stats.pt"),
            )

    # Write the metadata
    for subset, metadata in all_metadata.items():
        with open(
            dest.joinpath(f"{subset}.csv"), "w", encoding="utf-8"
        ) as file:
            file.write("\n".join(metadata))


class SpeechCommands(Dataset):
    """Creates a dataset for pre-processed Speech Commands."""

    SAMPLE_RATE: int = 16_000  # Speech Commands sample rate.
    NUM_CLASSES: int = 35  # Number of classes in Speech Commands

    def __init__(
        self,
        root: str,
        subset: str,
        build: bool = False,
        download: bool = True,
        transform: Optional[nn.Module] = None,
    ) -> None:
        """Create the dataset. If `build`, apply the `transform`
        to the waveforms to pre-process the data.
        The samples are then normalized according to statistics
        computed on the training set.

        Parameters
        ----------
        root : str
            Path to the directory where the dataset is found
            or to be built.
        subset : str
            Select a subset of the dataset. Must be in
            (`training`, `validation`, `testing`).
        build : bool, optional
            Whether to build the processed dataset from scratch
            or not, by default False.
        download : bool, optional
            Whether to download data for torchaudio SPEECHCOMMANDS,
            by default True.
        transform : Optional[nn.Module], optional
            Transform used to pre-process the input data.
            If it is not specified, the default transformation
            is to make log-compressed mel-spectrograms with 64 channels,
            computed with a window of 25 ms every 10 ms.
            By default None.
        """
        super().__init__()
        self.root = (
            Path(root).joinpath(f"{FOLDER_IN_ARCHIVE}/processed").resolve()
        )
        if subset not in ("training", "validation", "testing"):
            raise ValueError(f"Invalid subset {subset}.")
        self.subset = subset
        if build:  # pragma: no cover
            if transform is None:
                transform = default_audio_transform(self.SAMPLE_RATE)
            _build_dataset(
                root, self.root, self.SAMPLE_RATE, download, transform
            )

        stats = torch.load(self.root.joinpath("training_stats.pt"))
        self._mean, self._std = stats["mean"], stats["std"]

        self.metadata = {}
        with open(
            self.root.joinpath(f"{subset}.csv"), "r", encoding="utf-8"
        ) as file:
            for line in file.read().splitlines():
                idx, *other = line.split(",")
                self.metadata[int(idx)] = SpeechCommandsMetadata.from_csv(
                    idx, *other
                )

        self._path = self.root.joinpath(self.subset)
        self._n_digits = len(str(len(self.metadata)))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset. It is normalized
        according to the mean and standard deviation of the training set.

        Parameters
        ----------
        index : int
            Index of the sample

        Returns
        -------
        Tuple[torch.Tensor, int]
            Normalized features and label.
        """
        metadata = self.metadata[index]
        path = self._path.joinpath(f"{index:0{self._n_digits}d}.pt")
        normalized = (torch.load(path) - self._mean) / self._std
        return normalized, _LABELS[metadata.label]

    def __len__(self) -> int:
        return len(self.metadata)
