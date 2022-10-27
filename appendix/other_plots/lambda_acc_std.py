import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerTuple

NORM = "euclidean"

FIGSIZE = (10, 5)
LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 15
TICKS_FONTSIZE = 15
FILL_ALPHA = 0.1
BASELINE_COLOR = "orange"


def plot(lambd, mean, std, base_mean, base_std, use_val: bool):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(lambd, mean, std, fmt="o")
    hline = ax.hlines(
        base_mean,
        xmin=lambd.min(),
        xmax=lambd.max(),
        linestyles="dashed",
        colors=BASELINE_COLOR,
    )
    fill = ax.fill_between(
        lambd,
        base_mean - base_std,
        base_mean + base_std,
        alpha=FILL_ALPHA,
        color=BASELINE_COLOR,
    )

    ax.tick_params(axis="both", labelsize=TICKS_FONTSIZE)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(
        r"Validation accuracy" if use_val else r"Test accuracy",
        fontsize=LABEL_FONTSIZE,
    )
    ax.legend(
        [(hline, fill)],
        [r"Baseline mean $\pm$ std"],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=LEGEND_FONTSIZE,
    )
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

    for model, dataset in product(
        dataframe.model.unique(), dataframe.dataset.unique()
    ):
        base_df = dataframe.query(
            f"(model == '{model}') & "
            f"(~topographic) & "
            f"(dataset == '{dataset}')"
        )
        for position_scheme, dimension in product(
            dataframe.position_scheme.dropna().unique(),
            dataframe.dimension.dropna().unique(),
        ):

            subdf = dataframe.query(
                f"(model == '{model}') & "
                f"(dimension == {dimension}) & "
                f"(norm == '{NORM}') & "
                f"(position_scheme == '{position_scheme}') & "
                f"(dataset == '{dataset}')"
            )

            fig, ax = plot(
                subdf.lambd,
                subdf[f"{prefix}_mean"],
                subdf[f"{prefix}_std"],
                base_df[f"{prefix}_mean"],
                base_df[f"{prefix}_std"],
                args.use_val,
            )
            fig.savefig(f"{model}_{dataset}_{position_scheme}_{dimension}.pdf")
            plt.close()
