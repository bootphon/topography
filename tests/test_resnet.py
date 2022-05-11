from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from topography.models import resnet18
from topography.training import Writer, evaluate, train
from topography.utils import LinearWarmupCosineAnnealingLR


def test_resnet():
    temp_dir = TemporaryDirectory()
    lr, epochs = 1e-3, 2
    num_samples, num_classes, shape = 2, 10, [3, 32, 32]
    device = torch.device("cpu")
    model = resnet18()
    optimizer = SGD(model.parameters(), lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=1, max_epochs=epochs
    )
    writer = Writer(temp_dir.name)
    writer.log_hparams(lr=lr, epochs=epochs)
    data = torch.randn([num_samples] + shape)
    targets = torch.randint(num_classes, [num_samples])
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=num_samples)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        train(model, dataloader, optimizer, criterion, device, writer)
    evaluate(model, dataloader, criterion, device, writer, mode="test")
    writer.save(mode="test", metric="acc", maximize=True, model=model)
    writer.save(mode="test", metric="loss", maximize=False, model=model)
    scheduler.step()
    writer.close()
