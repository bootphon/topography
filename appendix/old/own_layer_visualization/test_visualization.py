from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from visualization import VisualizationLayerCNN


def test_visu_layer():
    temp_dir = TemporaryDirectory().name
    model = resnet18(pretrained=True).eval()

    layers = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]

    for l in layers:
        VisualizationLayerCNN(model, l, 0, n_iter=100)(seed=1).save(
            f"{temp_dir}/{l}.png"
        )

    nrows, ncols = 5, 4
    _, ax = plt.subplots(nrows, ncols, figsize=(20, 20))
    for k, l in enumerate(layers):
        img = Image.open(f"{temp_dir}/{l}.png")
        i, j = k // ncols, k % ncols
        ax[i, j].imshow(img)
        ax[i, j].set_title(l)
        ax[i, j].axis("off")
    plt.savefig(f"{temp_dir}/layers.pdf")
    plt.show()
