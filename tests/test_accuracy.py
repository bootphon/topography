"""Test of the custom accuracy function."""
import torch

from topography.training.training import accuracy


def test_accuracy():
    labels = torch.Tensor([0, 1, 1, 0])
    out1 = torch.Tensor([[0.8, 0.3], [0.1, 0.6], [0.2, 0.9], [10, -0.2]])
    out2 = torch.Tensor([[0.3, 0.8], [0.9, 0.6], [0.7, 0.3], [0.1, 0.2]])
    out3 = torch.Tensor([[100, 0.8], [-10, 0], [0.7, 0.3], [0.1, 0.2]])

    acc1 = accuracy(out1, labels)
    assert isinstance(acc1, float)
    assert acc1 == 1
    assert accuracy(out2, labels) == 0
    assert accuracy(out3, labels) == 0.5

    inp = torch.rand(10, 5)
    _, targets = torch.max(inp.data, 1)
    assert accuracy(inp, targets) == 1
    assert accuracy(inp, (targets + 1) % 5) == 0


if __name__ == "__main__":
    test_accuracy()
