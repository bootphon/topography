"""Plot and recap the results from pruning experiments."""
import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

from topography.utils import tensorboard_to_dataframe


def results_dataframe(prunedir: Path) -> pd.DataFrame:
    """Creates a recap dataframe.

    Parameters
    ----------
    prunedir : Path
        Directory containing all the pruning experiments.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the results.
    """
    dataframe = []
    for tb_root in prunedir.rglob("tensorboard"):
        try:
            df = tensorboard_to_dataframe(list(tb_root.glob("*"))[0])
        except:
            print(f"No tensorboard file in {tb_root}.")
            continue
        if len(df) == 0:
            print(f"Tensorboard file in {tb_root} empty.")
            continue
        with open(tb_root.parent / "environment/config.json", "r") as file:
            config = json.load(file)
        accuracy = float(df[df.metric.str.endswith("acc")].value)
        dataframe.append(
            [
                config["mode"],
                config["proportion"],
                config["ln_dimension"],
                config["seed"],
                config["topographic"],
                accuracy,
            ]
        )
    return pd.DataFrame(
        dataframe,
        columns=[
            "mode",
            "proportion",
            "ln_dimension",
            "seed",
            "topographic",
            "accuracy",
        ],
    ).drop_duplicates()


def plot_results(dataframe: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the results for one pruning configuration.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes.
    """
    topo_mean = dataframe.query("topographic").groupby("proportion").mean()
    topo_std = dataframe.query("topographic").groupby("proportion").std()
    base_mean = dataframe.query("~topographic").groupby("proportion").mean()
    base_std = dataframe.query("~topographic").groupby("proportion").std()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(
        topo_mean.index,
        topo_mean.accuracy - topo_std.accuracy,
        topo_mean.accuracy + topo_std.accuracy,
        alpha=0.2,
    )
    ax.plot(
        topo_mean.index,
        topo_mean.accuracy,
        "-",
        label="Topographic",
    )
    ax.fill_between(
        base_mean.index,
        base_mean.accuracy - base_std.accuracy,
        base_mean.accuracy + base_std.accuracy,
        alpha=0.2,
    )
    ax.plot(
        base_mean.index,
        base_mean.accuracy,
        "--",
        label="Baseline",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Percentage of channels pruned", fontsize=30)
    ax.set_ylabel("Test accuracy", fontsize=30)
    ax.tick_params("both", labelsize=20)
    ax.legend(loc="lower left", fontsize=25)
    fig
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prunedir", type=str, help="Pruning directory")
    parser.add_argument("--style", type=str, help="Matplotlib style file")

    args = parser.parse_args()
    if args.style is not None:
        plt.style.use(Path(args.style))
    prunedir = Path(args.prunedir).resolve()

    dataframe = results_dataframe(prunedir)
    dataframe.to_csv("prune_results.csv")

    fig, ax = plot_results(dataframe.query("mode == 'random'"))
    plt.savefig("random.pdf")
    plt.close()

    for ln_dimension in (1, 2):
        fig, ax = plot_results(
            dataframe.query(
                f"(mode == 'weight') & (ln_dimension == {ln_dimension})"
            )
        )
        plt.savefig(f"weight_{ln_dimension}.pdf")
        plt.close()
