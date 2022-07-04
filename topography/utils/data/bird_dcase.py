"""Provides the Bird DCASE dataset for birdsong detection, with
pre-processed features.
"""
import dataclasses
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torchaudio
from torch import nn
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import extract_archive
from torchaudio.functional import resample
from tqdm.auto import tqdm

from topography.utils import AverageMeter
from topography.utils.data.common import default_audio_transform

FOLDER_IN_ARCHIVE = "BirdDCASE"
_FILES = {
    "BirdVox-DCASE-20k": (
        "https://zenodo.org/api/files/4a8eaf84-3e69-4990-b5ff-fa0cc3fe4d24/BirdVox-DCASE-20k.zip",  # noqa: E501
        "https://ndownloader.figshare.com/files/10853300",
    ),
    "warblrb10k": (
        "https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip",  # noqa: E501
        "https://ndownloader.figshare.com/files/10853306",
    ),
    "freefield1010": (
        "https://archive.org/download/ff1010bird/ff1010bird_wav.zip",
        "https://ndownloader.figshare.com/files/10853303",
    ),
}


@dataclasses.dataclass
class BirdDCASEMetadata:
    """Metadata container.

    Attributes
    ----------
    idx : int
        Index of the sample in the current BirdDCASE dataset.
    itemid : str
        Id of the sample in its subset.
    datasetid : str
        Id of the subset. Either BirdVox-DCASE-20k, warblrb10k or freefield1010
    hasbird : int
        Label of the sample. 1 if it has a bird in it, else 0.
    """

    idx: int
    itemid: str
    datasetid: str
    hasbird: int

    def __post_init__(self):
        """Post-initialization: casts idx and hasbird as int."""
        self.idx = int(self.idx)
        self.hasbird = int(self.hasbird)


def download_bird_dcase(path: Path) -> None:
    """Download the BirdDCASE datset.

    Parameters
    ----------
    path : Path
        Destination path. The archives are downloaded in the parent
        directory.
        It will have the following structure:

        path/../
            ...
            BirdVox-DCASE-20k.zip
            warblrb10k_public_wav.zip
            freefield1010_wav.zip
            ...
            path/
            |--BirdVox-DCASE-20k/
            |----wav/
            |----labels.csv
            |
            |--warblrb10k/
            |----wav/
            |----labels.csv
            |
            |--freefield1010/
            |----wav/
            |----labels.csv
    """
    root = path.parent
    for dataset_name, (url, labels_url) in _FILES.items():
        dataset = path / dataset_name
        if not dataset.is_dir():
            archive = root / Path(url).name
            if not archive.is_file():
                download_url_to_file(url, archive)
            extract_archive(archive, dataset)
        labels = dataset / "labels.csv"
        if not labels.is_file():
            download_url_to_file(labels_url, labels)


def _process_dataset(
    path: Path, sample_rate: int, transform: nn.Module
) -> None:
    """Process the BirdDCASE dataset: extract features from each waveform.

    Parameters
    ----------
    path : Path
        Root path
    sample_rate : int
        Target sample rate. If an audio file has not the same sample rate,
        it will be resampled to this sample rate.
    transform : nn.Module
        Transformation to apply to the waveforms in order to extract
        features.
    """
    for src in path.glob("*/wav"):
        dataset = src.parent
        dest = dataset / "processed"
        dest.mkdir()
        mean, std = AverageMeter("mean"), AverageMeter("std")
        files = list(src.glob("*.wav"))
        for file in tqdm(files, leave=False, desc=f"Process {dataset.name}"):
            waveform, src_sample_rate = torchaudio.load(file)
            if src_sample_rate != sample_rate:
                waveform = resample(waveform, src_sample_rate, sample_rate)
            # Padding if necessary
            if waveform.shape[1] < sample_rate:
                temp = torch.zeros(1, sample_rate)
                temp[:, : waveform.shape[1]] = waveform
                waveform = temp

            feats = transform(waveform)
            mean.update(feats.mean())
            std.update(feats.std())
            torch.save(feats, (dest / file.stem).with_suffix(".pt"))

        torch.save(
            {"mean": mean.avg, "std": std.avg, "length": len(files)},
            dataset / "stats.pt",
        )


class BirdDCASE(Dataset):
    """BirdDCASE dataset. It is made of three datasets: BirdVox-DCASE-20k,
    warblrb10k and freefield1010.
    The data is pre-processed to use features instead of the raw audio.
    """

    SAMPLE_RATE: int = 16_000  # Target sample rate.

    def __init__(
        self,
        root: Union[str, Path],
        subset: str,
        download: bool = False,
        process: bool = False,
        validation_set: str = "freefield1010",
        transform: Optional[nn.Module] = None,
    ) -> None:
        """Creates the dataset.

        Parameters
        ----------
        root : Union[str, Path]
            Path to the directory where the dataset is found or downloaded.
            The dataset directory will have the following structure:

                root/BirdDCASE/
                |--BirdVox-DCASE-20k/
                |----wav/
                |----processed/
                |----stats.pt
                |----labels.csv
                |
                |--warblrb10k/
                |----wav/
                |----processed/
                |----stats.pt
                |----labels.csv
                |
                |--freefield1010/
                |----wav/
                |----processed/
                |----stats.pt
                |----labels.csv
        subset : str
            Select a subset of the dataset. Must be `training` or `validation`.
        download : bool, optional
            Whether to download the dataset if it is not found at root path,
            by default False.
        process : bool, optional
            Whether to process the dataset to extract the features,
            by default False.
        validation_set : str, optional
            The chosen validation set. Must be "BirdVox-DCASE-20k",
            "freefield1010" or "warblrb10k". By default "freefield1010"
        transform : Optional[nn.Module], optional
            Transform used to pre-process the input data.
            If it is not specified, the default transformation
            is to make log-compressed mel-spectrograms with 64 channels,
            computed with a window of 25 ms every 10 ms.
            By default None.

        Raises
        ------
        RuntimeError
            If the dataset if not found and `download` is set to False.
        ValueError
            If the validation set is not one of the three reference datasets
            or if the specified subset is not `training` or `validation`.
        """
        super().__init__()
        self._path = Path(root).resolve() / FOLDER_IN_ARCHIVE

        # Download the datasets
        if not self._path.is_dir():
            if download:
                download_bird_dcase(self._path)
            else:
                raise RuntimeError(
                    f"Dataset not found at {self._path}. "
                    + "Please set `download=True` to download the dataset."
                )

        # Process the datasets
        if process:
            if transform is None:
                transform = default_audio_transform(self.SAMPLE_RATE)
            _process_dataset(self._path, self.SAMPLE_RATE, transform)

        # Split validation / training
        if validation_set not in _FILES.keys():
            raise ValueError(
                f"Validation set {validation_set} is invalid:"
                f" must be in ({', '.join(_FILES.keys())})"
            )
        training_sets = sorted(list(_FILES.keys()))
        training_sets.remove(validation_set)

        # Build metadata
        self.metadata: Dict[int, BirdDCASEMetadata] = {}

        if subset == "validation":
            datasets = [validation_set]
        elif subset == "training":
            datasets = training_sets
        else:
            raise ValueError(f"Invalid subset {subset}.")

        idx = 0
        for dataset in datasets:
            with open(
                self._path / dataset / "labels.csv", "r", encoding="utf-8"
            ) as file:
                lines = file.read().splitlines()[1:]
            for line in lines:
                itemid, datasetid, hasbird = line.split(",")
                self.metadata[idx] = BirdDCASEMetadata(
                    idx, itemid, datasetid, hasbird
                )
                idx += 1

        # Mean and std
        stats = [
            torch.load(self._path / dataset / "stats.pt")
            for dataset in training_sets
        ]
        length = sum(stat["length"] for stat in stats)
        self._mean = (
            sum(stat["mean"] * stat["length"] for stat in stats) / length
        )
        self._std = sum(stat["std"] * stat["length"] for stat in stats) / length

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
        path = (
            self._path
            / metadata.datasetid
            / "processed"
            / (metadata.itemid + ".pt")
        )
        normalized = (torch.load(path) - self._mean) / self._std
        return normalized, metadata.hasbird

    def __len__(self) -> int:
        return len(self.metadata)
