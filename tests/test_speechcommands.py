"""Test of the Speech VGG on Speech Commands."""
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchaudio import transforms
from torchaudio.datasets.speechcommands import FOLDER_IN_ARCHIVE

from topography.models import vgg16_bn
from topography.training import Writer, evaluate, train
from topography.training.training import accuracy
from topography.utils import LinearWarmupCosineAnnealingLR
from topography.utils.data import SpeechCommands
from topography.utils.data import speechcommands as spc
from topography.utils.data.common import default_audio_transform


def test_labels():
    assert isinstance(spc._LABELS, dict)
    assert len(spc._LABELS) == 35
    assert set(spc._LABELS.values()) == set(range(35))


def test_metadata():
    metadata = spc.SpeechCommandsMetadata(0, 16_000, "dog", "abc", 2)
    csv_line = "0,16000,dog,abc,2"
    assert str(metadata) == csv_line
    splitted_line = "0,16000,dog,abc,2".split(",")
    assert metadata == spc.SpeechCommandsMetadata.from_csv(*splitted_line)


def test_default_transform():
    sample_rate, n_mels = 16_000, 64
    window_duration, hop_duration = 0.025, 0.01
    transform = default_audio_transform(sample_rate=sample_rate)
    assert isinstance(transform, nn.Module)
    mel_spectrogram, ampl_to_db = transform[0], transform[1]
    assert isinstance(mel_spectrogram, transforms.MelSpectrogram)
    assert isinstance(ampl_to_db, transforms.AmplitudeToDB)

    assert mel_spectrogram.sample_rate == sample_rate
    assert (
        mel_spectrogram.n_fft
        == mel_spectrogram.win_length
        == int(window_duration * sample_rate)
    )
    assert mel_spectrogram.hop_length == int(hop_duration * sample_rate)
    assert mel_spectrogram.n_mels == n_mels

    assert ampl_to_db.stype == "power"


def test_bad_subset(tmp_path):
    with pytest.raises(ValueError) as error:
        SpeechCommands(tmp_path, subset="bad")
    assert str(error.value).startswith("Invalid subset")


def test_vgg16_bn(tmp_path):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(0)

    root = Path(f"{tmp_path}/{FOLDER_IN_ARCHIVE}/processed").resolve()
    root.mkdir(parents=True)
    root.joinpath("training").mkdir()

    sample_rate, duration, num_samples = 16_000, 1, 4
    lr, epochs = 0.001, 30
    num_classes, n_digits = 5, len(str(num_samples))

    for idx in range(num_samples):
        waveform = torch.rand(1, sample_rate * duration)
        feats = default_audio_transform(sample_rate)(waveform)
        torch.save(feats, root.joinpath(f"training/{idx:0{n_digits}d}.pt"))

    stats = {"mean": torch.tensor(0), "std": torch.tensor(1)}
    torch.save(stats, root.joinpath("training_stats.pt"))

    targets = torch.arange(num_samples)
    inv_labels = {value: key for key, value in spc._LABELS.items()}
    labels = [inv_labels[target.item()] for target in targets]
    metadata = [
        str(spc.SpeechCommandsMetadata(idx, sample_rate, label, "dummy", 0))
        for idx, label in zip(range(num_samples), labels)
    ]
    with open(root.joinpath("training.csv"), "w") as file:
        file.write("\n".join(metadata))

    dataset = SpeechCommands(root=tmp_path, subset="training", build=False)
    with pytest.raises(IndexError):
        dataset[num_samples + 1]

    device = torch.device("cpu")
    model = vgg16_bn(num_classes=num_classes, in_channels=1)
    optimizer = SGD(model.parameters(), lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=1, max_epochs=epochs
    )
    writer = Writer(tmp_path / "writer", backup_setup=False)
    writer.log_config(dict(lr=lr, epochs=epochs))
    dataloader = DataLoader(dataset, batch_size=num_samples, generator=g)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        train(
            model,
            dataloader,
            optimizer,
            criterion,
            device,
            writer,
            is_pytorch_loss=True,
        )
        scheduler.step()
    evaluate(
        model,
        dataloader,
        criterion,
        device,
        writer,
        mode="test",
        is_pytorch_loss=True,
    )

    model.eval()
    batch = next(iter(dataloader))
    assert accuracy(model(batch[0]), batch[1]).value == 1.0
    writer.close()
