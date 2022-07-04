"""Utilites shared between custom datasets."""
from torch import nn
from torchaudio import transforms


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
