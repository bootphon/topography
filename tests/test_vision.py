"""Test of the ResNet. Checks if it can overfit a single batch."""
from tempfile import TemporaryDirectory

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from topography import MetricOutput, TopographicLoss, TopographicModel, models
from topography.training import Writer, evaluate, train
from topography.training.training import accuracy
from topography.utils import LinearWarmupCosineAnnealingLR


@pytest.mark.parametrize("model_name", ["resnet18", "vgg16_bn", "densenet121"])
def test_vision(model_name):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    temp_dir = TemporaryDirectory()
    g = torch.Generator()
    g.manual_seed(0)
    lr, epochs = 0.01, 30
    cifar_classes, num_samples, shape = 10, 4, [3, 32, 32]
    device = torch.device("cpu")
    model = getattr(models, model_name)(cifar_classes)
    optimizer = SGD(model.parameters(), lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=0.3 * epochs, max_epochs=epochs
    )
    writer = Writer(temp_dir.name)
    writer.log_config(dict(lr=lr, epochs=epochs))
    data = torch.randn([num_samples] + shape)
    targets = torch.randint(cifar_classes, [num_samples])
    dataset = TensorDataset(data, targets)
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
    output = model(data)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (num_samples, cifar_classes)
    assert accuracy(output, targets).value == 1.0

    writer.save(mode="test", metric="acc", maximize=True, model=model)
    writer.save(mode="test", metric="loss", maximize=False, model=model)
    writer.close()


@pytest.mark.parametrize("model_name", ["resnet18", "vgg16_bn", "densenet121"])
def test_vision_topographic(model_name):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    temp_dir = TemporaryDirectory()
    g = torch.Generator()
    g.manual_seed(0)
    lr, epochs = 0.01, 30
    cifar_classes, num_samples, shape = 10, 4, [3, 32, 32]
    device = torch.device("cpu")
    model = TopographicModel(
        getattr(models, model_name)(cifar_classes),
        topographic_layer_names=models.topographic_layer_names(model_name),
    )
    optimizer = SGD(model.parameters(), lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=0.3 * epochs, max_epochs=epochs
    )
    writer = Writer(temp_dir.name)
    writer.log_config(dict(lr=lr, epochs=epochs))
    data = torch.randn([num_samples] + shape)
    targets = torch.randint(cifar_classes, [num_samples])
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=num_samples, generator=g)
    cross_entropy = nn.CrossEntropyLoss()
    topo_loss = TopographicLoss()
    lambd = 0.1

    def criterion(output, target):
        ce = cross_entropy(output, target)
        topo = topo_loss(model.activations, model.inverse_distance)
        return MetricOutput(
            value=ce + lambd * topo.value,
            extras={
                "loss-cross-entropy": ce.item(),
                "loss-topographic": topo.value.item(),
                **topo.extras,
            },
        )

    for _ in range(epochs):
        train(model, dataloader, optimizer, criterion, device, writer)
        scheduler.step()
    evaluate(model, dataloader, criterion, device, writer, mode="test")

    model.eval()
    output = model(data)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (num_samples, cifar_classes)
    assert accuracy(output, targets).value == 1.0

    writer.save(mode="test", metric="acc", maximize=True, model=model)
    writer.save(mode="test", metric="loss", maximize=False, model=model)
    writer.close()
