import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INDICES = {
    (2.0, "hypercube"): "2D grid",
    (2.0, "hypersphere"): "Circle",
    (2.0, "nested"): "2D nested",
    (3.0, "hypercube"): "3D grid",
    (3.0, "hypersphere"): "Sphere",
    (3.0, "nested"): "3D nested",
}

MODELS = {
    "resnet18": "ResNet-18",
    "densenet121": "DenseNet-121",
    "vgg16_bn": "VGG-16",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recap", type=str)
    parser.add_argument("--style", type=str)

    args = parser.parse_args()
    if args.style is not None:
        plt.style.use(Path(args.style).resolve())

    alldf = pd.read_csv(Path(args.recap).resolve(), index_col=0)
    for dataset in alldf.dataset.unique():
        fig, ax = plt.subplots(figsize=(12, 5))

        for model, marker in zip(
            ["densenet121", "resnet18", "vgg16_bn"],
            ["o", "*", "D"],
        ):
            df = alldf.query(
                f"(model == '{model}') & (dataset == '{dataset}') & "
                f"(topographic) & (seed == 0) & (lambd == 1)"
            )

            scales = {
                (row.dimension, row.position_scheme): row.scale
                for _, row in df.iterrows()
            }

            plt.scatter(
                range(len(df)),
                [scales[k] for k in INDICES.keys()],
                marker=marker,
                s=100,
                label=MODELS[model],
            )

        plt.xticks(
            range(len(df)),
            INDICES.values(),
        )

        plt.xlabel(r"Topographic organization", fontsize=20)
        plt.ylabel(
            r"$\mathcal{L}_\text{topo} / \mathcal{L}_\text{classif}$",
            fontsize=20,
        )
        ax.tick_params("x", labelsize=15)
        ax.tick_params("y", labelsize=15)
        plt.legend(fontsize=15)
        plt.savefig(f"lambda_scale_{dataset}.pdf")
        plt.close()
