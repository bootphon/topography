"""Test of the ResNet. Checks if it can overfit a single batch."""
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from topography.models import resnet18
from topography.training import Writer, evaluate, train
from topography.training.training import accuracy
from topography.utils import LinearWarmupCosineAnnealingLR


def test_resnet():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    temp_dir = TemporaryDirectory()
    g = torch.Generator()
    g.manual_seed(0)
    lr, epochs = 0.1, 10
    cifar_classes, num_samples, shape = 10, 4, [3, 32, 32]
    device = torch.device("cpu")
    model = resnet18()
    optimizer = SGD(model.parameters(), lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=1, max_epochs=epochs
    )
    writer = Writer(temp_dir.name)
    writer.log_hparams(lr=lr, epochs=epochs)
    data = torch.randn([num_samples] + shape)
    targets = torch.randint(cifar_classes, [num_samples])
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=num_samples, generator=g)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        train(model, dataloader, optimizer, criterion, device, writer)
        scheduler.step()
    evaluate(model, dataloader, criterion, device, writer, mode="test")

    model.eval()
    output = model(data)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (num_samples, cifar_classes)
    assert accuracy(output, targets) == 1.0

    writer.save(mode="test", metric="acc", maximize=True, model=model)
    writer.save(mode="test", metric="loss", maximize=False, model=model)
    writer.close()


if __name__ == "__main__":
    test_resnet()
