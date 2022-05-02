import torch.nn as nn
from torchvision import models


def resnet(out_features: int = 1000, pretrained: bool = True,
           model_size: int = 18) -> models.ResNet:
    if model_size not in [18, 34, 50, 101, 152]:
        raise ValueError('Invalid resnet model.')
    model = getattr(models, f'resnet{model_size}')(pretrained=pretrained)
    if out_features != model.fc.out_features:
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=out_features)
    return model
