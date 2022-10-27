import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NORM = "euclidean"

FIGSIZE = (10, 5)
LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 15

INDICES = {
    (2.0, "hypercube"): "2D grid",
    (2.0, "hypersphere"): "Circle",
    (2.0, "nested"): "2D nested",
    (3.0, "hypercube"): "3D grid",
    (3.0, "hypersphere"): "Sphere",
    (3.0, "nested"): "3D nested",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--style")
    parser.add_argument("--use_val", action="store_true")
    args = parser.parse_args()

    dataframe = pd.read_csv(Path(args.csv).resolve(), index_col=0)
    if args.style is not None:
        plt.style.use(Path(args.style).resolve())

    prefix = "val_acc" if args.use_val else "test_acc"

    for model, dataset in product(
        dataframe.model.unique(), dataframe.dataset.unique()
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for position_scheme, dimension in product(
            dataframe.position_scheme.dropna().unique(),
            dataframe.dimension.dropna().unique(),
        ):

            subdf = dataframe.query(
                f"(model == '{model}') & "
                f"(((norm == '{NORM}') & "
                f"(position_scheme == '{position_scheme}') & "
                f"(dimension == {dimension})) | (~topographic)) & "
                f"(dataset == '{dataset}')"
            )

            ax.plot(
                subdf.lambd,
                subdf[f"{prefix}_mean"],
                label=INDICES.get((dimension, position_scheme), "Baseline"),
            )

        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(
            r"Validation accuracy" if args.use_val else r"Test accuracy",
            fontsize=LABEL_FONTSIZE,
        )
        ax.legend(fontsize=LEGEND_FONTSIZE)
        plt.close()
