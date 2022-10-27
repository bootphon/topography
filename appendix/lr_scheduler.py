import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import SGD

from topography.utils import LinearWarmupCosineAnnealingLR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", help="Path to mplstyle file.")
    parser.add_argument("--prop", type=float, default=0.3)
    args = parser.parse_args()

    if args.style is not None:
        plt.style.use(Path(args.style).resolve())

    lr, epochs = 1, 10000
    param = torch.ones(10, requires_grad=True)
    optim = SGD([param], lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(optim, args.prop * epochs, epochs)

    steps = []
    for _ in range(epochs):
        optim.step()
        scheduler.step()
        steps += scheduler.get_last_lr()

    plt.figure(figsize=(4, 2))
    plt.plot(range(epochs), steps)
    plt.xticks(
        [0, args.prop * epochs, epochs],
        [0, r"$0.3\times \text{epochs}$", "epochs"],
    )
    plt.yticks([0, 0.5 * lr, lr], [0, r"$\frac{\eta}{2}$", r"$\eta$"])
    plt.savefig("lr_scheduler.pdf")
    plt.close()
