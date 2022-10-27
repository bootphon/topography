import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NORM = "euclidean"

FIGSIZE = (10, 5)
LABEL_FONTSIZE = 20
TICKS_FONTSIZE = 15
BASELINE_COLOR = "orange"
TOPO_COLOR = "blue"

INDICES = {
    (2.0, "hypercube"): "2D grid",
    (2.0, "hypersphere"): "Circle",
    (2.0, "nested"): "2D nested",
    (3.0, "hypercube"): "3D grid",
    (3.0, "hypersphere"): "Sphere",
    (3.0, "nested"): "3D nested",
}


def plot(topo_orga, mean, std, use_val: bool):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    baseline_errorbar = ax.errorbar(
        topo_orga.iloc[0],
        mean.iloc[0],
        std.iloc[0],
        fmt="s",
        color=BASELINE_COLOR,
    )
    baseline_errorbar[-1][0].set_linestyle("--")
    ax.errorbar(
        topo_orga.iloc[1:],
        mean.iloc[1:],
        std.iloc[1:],
        fmt="o",
        color=TOPO_COLOR,
    )

    ax.tick_params(axis="both", labelsize=TICKS_FONTSIZE)
    ax.set_ylabel(
        r"Validation accuracy" if use_val else r"Test accuracy",
        fontsize=LABEL_FONTSIZE,
    )
    ax.set_xlabel(r"Topographic organization", fontsize=LABEL_FONTSIZE)
    return fig, ax


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

    for model, dataset, lambd in product(
        dataframe.model.unique(),
        dataframe.dataset.unique(),
        dataframe.lambd.dropna().unique(),
    ):
        subdf = dataframe.query(
            f"(model == '{model}') & "
            f"(((norm == '{NORM}') & (lambd == {lambd})) | (~topographic)) "
            f"& (dataset == '{dataset}')"
        )
        fig, ax = plot(
            subdf[["dimension", "position_scheme"]].apply(
                lambda x: INDICES.get(tuple(x), "Baseline"), axis=1
            ),
            subdf[f"{prefix}_mean"],
            subdf[f"{prefix}_std"],
            args.use_val,
        )
        fig.savefig(f"topo_orga/{model}_{dataset}_{lambd}.pdf")
        plt.close()
