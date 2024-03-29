"""Provides the Bird DCASE dataset for birdsong detection, with
pre-processed features.
"""
import dataclasses
import math
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
from topography.utils.data import common

BirdDCASEUrls = Dict[str, Tuple[str, str]]

URLS: BirdDCASEUrls = {
    "BirdVox-DCASE-20k": (
        "https://zenodo.org/api/files/4a8eaf84-3e69-4990-b5ff-fa0cc3fe4d24/BirdVox-DCASE-20k.zip",  # pylint: disable=line-too-long # noqa: E501
        "https://ndownloader.figshare.com/files/10853300",
    ),
    "warblrb10k": (
        "https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip",  # pylint: disable=line-too-long # noqa: E501
        "https://ndownloader.figshare.com/files/10853306",
    ),
    "ff1010bird": (
        "https://archive.org/download/ff1010bird/ff1010bird_wav.zip",
        "https://ndownloader.figshare.com/files/10853303",
    ),
}

FOLDER_IN_ARCHIVE = "BirdDCASE"
SUBSETS = ["training", "validation", "testing"]
EVALUATION_SUBSETS = ["validation", "testing"]


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
        Id of the subset. Either BirdVox-DCASE-20k, warblrb10k or ff1010bird
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


def download_bird_dcase(
    path: Path, urls: BirdDCASEUrls
) -> None:  # pragma: no cover
    """Downloads the BirdDCASE datset.

    Parameters
    ----------
    path : Path
        Destination path. The archives are downloaded in the parent
        directory. It will have the following structure:
        path/../
            ...
            BirdVox-DCASE-20k.zip
            warblrb10k_public_wav.zip
            ff1010bird_wav.zip
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
            |--ff1010bird/
            |----wav/
            |----labels.csv
    urls: BirdDCASEUrls
        URLs of the waveforms and labels.
        Dictionary associating the intermediate dataset name
        to a tuple containing the URL of the waveforms and the URL
        of the labels.
    """
    root = path.parent
    for dataset_name, (url, labels_url) in urls.items():
        dataset = path / dataset_name
        if not dataset.is_dir():
            archive = root / Path(url).name
            if not archive.is_file():
                download_url_to_file(url, archive)
            extract_archive(archive, dataset)
        labels = dataset / "labels.csv"
        if not labels.is_file():
            download_url_to_file(labels_url, labels)


def _assign_subset(
    num_samples: int, split: Tuple[float, float], generator: torch.Generator
) -> Dict[int, str]:
    """Creates a dictionary assigning for each sample the subset
    it is assigned to ("training", "validation" or "testing").

    Parameters
    ----------
    num_samples : int
        Number of samples in the dataset.
    split : Tuple[float, float]
        Proportion of training and validation data in the dataset.
        Each value must be >0 and <1. The testing set will be made
        of the rest of the samples.
    generator : torch.Generator
        PyTorch pseudo random number generator.

    Returns
    -------
    Dict[int, str]
        Dictionary assigning for each sample index the corresponding subset.
    """
    indices = torch.randperm(num_samples, generator=generator).numpy()
    lengths = [math.floor(prop * num_samples) for prop in split]
    lengths.append(num_samples - sum(lengths))
    assign = sum(
        ([subset] * length for subset, length in zip(SUBSETS, lengths)), []
    )
    return dict(zip(indices, assign))


def _compute_subsets_statistics(
    path: Path, split: Tuple[float, float], seed: int
) -> Dict[str, Union[int, float, torch.Tensor]]:
    """Computes the dataset mean and standard deviation of the processed
    features, and splits each intermediate dataset ("BirdVox-DCASE-20k",
    "warblrb10k" and "ff1010bird") into training, validation and testing.

    Parameters
    ----------
    path : Path
        Root path.
    split : Tuple[float, float]
        Proportion of training and validation data in the dataset.
        Each value must be >0 and <1. The testing set will be made
        of the rest of the samples.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, Union[int, float, torch.Tensor]]:
        Dataset statistics: mean, standard deviation, split proportions and
        random seed.
    """
    generator = torch.Generator().manual_seed(seed)
    mean, std = AverageMeter("mean"), AverageMeter("std")
    for src in path.glob("*/processed"):
        subsets = ["itemid,subset"]
        files = sorted(list(src.glob("*.pt")))
        assignements = _assign_subset(len(files), split, generator)
        for idx, file in enumerate(files):
            feats = torch.load(file)
            if assignements[idx] == "training":
                mean.update(feats.mean())
                std.update(feats.std())
            subsets.append(f"{file.stem},{assignements[idx]}")
        with open(src.parent / "split.csv", "w", encoding="utf-8") as file:
            file.write("\n".join(subsets))
    stats = {"mean": mean.avg, "std": std.avg, "split": split, "seed": seed}
    torch.save(stats, path / "stats.pt")
    return stats


def _process_dataset(
    path: Path,
    sample_rate: int,
    process_fn: nn.Module,
) -> None:
    """Process the BirdDCASE dataset: extract features from each waveform.

    Parameters
    ----------
    path : Path
        Root path.
    sample_rate : int
        Target sample rate. If an audio file has not the same sample rate,
        it will be resampled to this sample rate.
    process_fn : nn.Module
        Transformation to apply to the waveforms in order to extract
        features.
    """
    for src in path.glob("*/wav"):
        dataset = src.parent
        dest = dataset / "processed"
        dest.mkdir(exist_ok=True)
        files = list(src.glob("*.wav"))
        for file in tqdm(files, leave=False, desc=f"Process {dataset.name}"):
            audio, src_sr = torchaudio.load(file)  # pylint: disable=no-member
            if src_sr != sample_rate:
                waveform = resample(audio, src_sr, sample_rate)
            feats = process_fn(waveform)
            torch.save(feats, (dest / file.stem).with_suffix(".pt"))


def _build_metadata(
    path: Path, subset: str, urls: BirdDCASEUrls
) -> Dict[int, BirdDCASEMetadata]:
    """Build the metadata dictionnary for the considered subset.

    Parameters
    ----------
    path : Path
        Root path.
    subset: str
        Considered subset, either "training", "validation" or "testing".
    urls: BirdDCASEUrls
        URLs of the waveforms and labels.
        Dictionary associating the intermediate dataset name
        to a tuple containing the URL of the waveforms and the URL
        of the labels.

    Returns
    -------
    Dict[int, BirdDCASEMetadata]
        Metadata dictionnary.
    """
    metadata, idx = {}, 0
    for dataset in sorted(urls.keys()):
        with open(path / dataset / "split.csv", "r", encoding="utf-8") as file:
            split = dict(
                [line.split(",") for line in file.read().splitlines()[1:]]
            )
        with open(path / dataset / "labels.csv", "r", encoding="utf-8") as file:
            lines = file.read().splitlines()[1:]
        for line in lines:
            line_metadata = BirdDCASEMetadata(idx, *line.split(","))
            if split[line_metadata.itemid] == subset:
                metadata[idx] = line_metadata
                idx += 1
    return metadata


class BirdDCASE(Dataset):
    """BirdDCASE dataset. It is made of three datasets: BirdVox-DCASE-20k,
    warblrb10k and ff1010bird.
    The data is pre-processed to use features instead of the raw audio.
    """

    SAMPLE_RATE: int = 16_000  # Target sample rate.

    def __init__(
        self,
        root: Union[str, Path],
        subset: str,
        *,
        download: bool = False,
        process: bool = False,
        process_fn: Optional[nn.Module] = None,
        split: Tuple[float, float] = (0.8, 0.1),
        duration: int = 1,
        urls: Optional[BirdDCASEUrls] = None,
        seed: int = 0,
        **kwargs,
    ) -> None:
        """Creates the dataset.

        Parameters
        ----------
        root : Union[str, Path]
            Path to the directory where the dataset is found or downloaded.
            The dataset directory will have the following structure:
                root/
                |--BirdVox-DCASE-20k.zip
                |--warblrb10k_public_wav.zip
                |--ff1010bird_wav.zip
                |
                |--stats.pt
                |--BirdDCASE/
                |----BirdVox-DCASE-20k/
                |------wav/
                |------processed/
                |------labels.csv
                |------split.csv
                |
                |----warblrb10k/
                |------wav/
                |------processed/
                |------labels.csv
                |------split.csv
                |
                |----ff1010bird/
                |------wav/
                |------processed/
                |------labels.csv
                |------split.csv
        subset : str
            Select a subset of the dataset. Must be `training`, `validation`
            or `testing`.
        download : bool, optional
            Whether to download the dataset if it is not found at root path,
            by default False.
        process : bool, optional
            Whether to process the dataset to extract the features,
            by default False.
        process_fn : Optional[nn.Module], optional
            Transform used to pre-process the input data.
            If it is not specified, the default transformation
            is to make log-compressed mel-spectrograms with 64 channels,
            computed with a window of 25 ms every 10 ms.
            By default None.
        split : Tuple[float, float]
            Proportion of training and validation data in the dataset.
            Each value must be >0 and <1. The testing set will be made
            of the rest of the samples.
        duration : int, optional
            Duration in seconds of the audio segment corresponding to
            the croped features. Used only for training. By default 1.
        urls: BirdDCASEUrls, optional
            URLs of the waveforms and labels.
            Dictionary associating the intermediate dataset name
            to a tuple containing the URL of the waveforms and the URL
            of the labels. By default None.
        **kwargs :
            torchvision.transforms.RandomCrop kwargs.

        Raises
        ------
        RuntimeError
            If the dataset if not found and `download` is set to False, or
            if the number of audio files is not the same as the number
            of processed features for one of the intermediate dataset.
        ValueError
            If the subset is not `training`, `validation` or `testing`.
        """
        super().__init__()
        if subset not in SUBSETS:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be in {SUBSETS}."
            )
        if urls is None:
            urls = URLS
        self._path = Path(root).resolve() / FOLDER_IN_ARCHIVE

        # Download the datasets
        if not self._path.is_dir():
            if download:
                download_bird_dcase(self._path, urls)
            else:
                raise RuntimeError(
                    f"Dataset not found at {self._path}. "
                    + "Please set `download=True` to download the dataset."
                )

        # Process the datasets
        if process_fn is None:
            process_fn = common.default_audio_transform(self.SAMPLE_RATE)
        if process:
            _process_dataset(self._path, self.SAMPLE_RATE, process_fn)
        else:
            for dataset in urls.keys():
                dataset_path = self._path / dataset
                if len(list((dataset_path / "wav").glob("*.wav"))) != len(
                    list((dataset_path / "processed").glob("*.pt"))
                ):
                    raise RuntimeError(
                        f"The number of audio files is not the same as "
                        f"the number of processed files in dataset {dataset}. "
                        f"Please set `process=True`."
                    )

        # Mean and std; split into subsets
        if not (self._path / "stats.pt").is_file():
            stats = _compute_subsets_statistics(self._path, split, seed)
        else:
            stats = torch.load(self._path / "stats.pt")
            if stats["split"] != split or stats["seed"] != seed:
                stats = _compute_subsets_statistics(self._path, split, seed)
        self._mean, self._std = stats["mean"], stats["std"]

        # Build metadata
        self.metadata = _build_metadata(self._path, subset, urls)

        # Crop or not
        self.crop = subset not in EVALUATION_SUBSETS
        self.transform = (
            common.RandomAudioFeaturesCrop(
                self.SAMPLE_RATE,
                duration=duration,
                transform=process_fn,
                pad_if_needed=True,
                **kwargs,
            )
            if self.crop
            else lambda inp: inp
        )

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
        if index >= len(self):
            raise IndexError
        metadata = self.metadata[index]
        path = (
            self._path
            / metadata.datasetid
            / "processed"
            / (metadata.itemid + ".pt")
        )
        normalized = (torch.load(path) - self._mean) / self._std
        return self.transform(normalized), metadata.hasbird

    def __len__(self) -> int:
        return len(self.metadata)
