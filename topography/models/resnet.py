"""Provides ResNet18 model."""
from topography.utils.externals.resnet import BasicBlock, ResNet


def resnet18(**kwargs):
    """Constructs a ResNet-18."""
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
