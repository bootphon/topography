"""Utilites shared between custom datasets."""
from typing import Optional

import torch
from torch import nn
from torch.utils.data import Dataset
from torchaudio import transforms
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as F

from topography.base import Metric
from topography.training import Writer


def default_audio_transform(
    sample_rate: int,
    window_duration: int = 25e-3,
    hop_duration: int = 10e-3,
    n_mels: int = 64,
) -> nn.Module:
    """Default transformation on waveforms for audio datasets: returns
    log-compressed mel-spectrograms with, by default, 64 channels,
    computed with a window of 25 ms every 10 ms.

    Parameters
    ----------
    sample_rate : int
        Sample rate of audio signal.
    window_duration : int, optional
        Duration of each window in seconds, by default 25e-3.
    hop_duration : int, optional
        Duration between successive windows, by default 10e-3.
    n_mels : int, optional
        Number of mel filterbanks., by default 64.

    Returns
    -------
    nn.Module
        Default transformation.
    """
    return nn.Sequential(
        transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(window_duration * sample_rate),
            hop_length=int(hop_duration * sample_rate),
            n_mels=n_mels,
        ),
        transforms.AmplitudeToDB(),
    )


class RandomAudioFeaturesCrop(RandomCrop):
    """Crop audio features at a random location."""

    def __init__(
        self,
        sample_rate: int,
        transform: Optional[nn.Module] = None,
        duration: int = 1,
        **kwargs,
    ):
        """Creates the module to crop audio features.
        If, for example, the `transform` computes mel-spectrograms
        with a sample rate of 16_000, setting `num_samples` to 16_000
        will crop the features to a segment corresponding to 1 second of audio.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the audio signal.
        transform : Optional[nn.Module], optional
            Transform used to pre-process the input data.
            If it is not specified, the default transformation
            is to make log-compressed mel-spectrograms with 64 channels,
            computed with a window of 25 ms every 10 ms.
            By default None.
        duration : int
            Duration in seconds of the audio segment corresponding to
            the croped features.
        **kwargs :
            torchvision.transforms.RandomCrop kwargs.
        """
        super().__init__((1, 1), **kwargs)
        sample = torch.rand(sample_rate * duration)
        if transform is None:
            transform = default_audio_transform(sample_rate)
        self.size = transform(sample).shape


def evaluate_with_crop(
    model: nn.Module,
    dataset: Dataset,
    metric: Metric,
    device: torch.device,
    writer: Writer,
    *,
    sample_rate: int,
    transform: Optional[nn.Module] = None,
    duration: int = 1,
) -> float:
    model.eval()
    writer.next_epoch()
    if transform is None:
        transform = default_audio_transform(sample_rate)
    height, width = transform(torch.rand(sample_rate * duration)).shape

    with torch.no_grad():
        for idx, (sample, target) in enumerate(dataset):
            data = torch.cat(
                [
                    F.crop(sample, 0, left, height, width)
                    for left in range(0, sample.shape[-1], width)
                ]
            )
            output = model(data).to(device)
            score = metric(output, target)
