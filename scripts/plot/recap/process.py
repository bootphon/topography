import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topography.utils import tensorboard_to_dataframe


def plot_processed_recap(
    dataframe: pd.DataFrame, path: Path, overwrite: bool
) -> None:
    """Process the recap dataframe. Plot the test accuracy as a function
    of lambda, for the different models and dimension of the positions.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe containing the recap of the experiments.
    path : Path
        Path to save the recap plots.
    overwrite : bool
        Whether to overwrite existing files or not.
    """
    if path.exists() and not overwrite:
        return
    prod = [
        (model, dataset)
        for model in dataframe.model.unique()
        for dataset in dataframe[dataframe.model == model].dataset.unique()
    ]
    nrows = int(np.sqrt(len(prod)))
    ncols = len(prod) // nrows
    ncols += 1 if ncols * nrows < len(prod) else 0
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    for k, (model, dataset) in enumerate(prod):
        i, j = k % nrows, k // nrows

        df_model = dataframe[
            (dataframe.model == model)
            & (dataframe.dataset == dataset)
            & dataframe.topographic
        ]
        reference = dataframe[
            (dataframe.model == model)
            & (dataframe.dataset == dataset)
            & ~dataframe.topographic
        ].test_acc.mean()
        if not np.isnan(reference):
            ax[i, j].hlines(
                reference,
                df_model.lambd.min(),
                df_model.lambd.max(),
                label="ref",
                colors="k",
                zorder=1,
            )

        for dim in sorted(df_model.dimension.unique()):
            subdf = df_model[df_model.dimension == dim]
            ax[i, j].scatter(
                subdf.lambd,
                subdf.test_acc,
                label=f"dim={int(dim)}",
                alpha=0.3,
                edgecolors="none",
                linewidths=2,
                s=50,
                zorder=2,
            )
        ax[i, j].legend(loc="lower left")
        ax[i, j].set_title(f"{model}, {dataset}")
        ax[i, j].set_xscale("log")
        ax[i, j].set_xlabel("lambda")
        ax[i, j].set_ylabel("Test acc")
    fig.savefig(path)
    plt.close()


def add_columns_to_recap(
    dataframe: pd.DataFrame, path: Path, overwrite: bool
) -> None:
    if path.exists() and not overwrite:
        return
    scale = []
    for _, row in dataframe.iterrows():
        tb_dir = Path(row.log).joinpath("tensorboard")
        if not tb_dir.exists():
            raise ValueError(
                f"Current row has a non valid 'log' entry: {row.log}"
            )
        if len(list(tb_dir.glob("*"))) != 1:
            raise ValueError(
                "Tensorboard directory does not have only one "
                + f"element, but {len(list(tb_dir.glob('*')))}: {tb_dir}."
            )
        tb_df = tensorboard_to_dataframe(str(next(tb_dir.glob("*"))))

        # Add column with lambda*topo_loss / cross_entropy
        if row.topographic:
            subdf = tb_df[tb_df.step == 1.0]
            cross_entropy_loss = subdf[
                subdf.metric == "train/extras/loss-cross-entropy"
            ].value.to_numpy()[0]
            topo_loss = subdf[
                subdf.metric == "train/extras/loss-topographic"
            ].value.to_numpy()[0]
            scale.append(f"{row.lambd * topo_loss / cross_entropy_loss:0.3e}")
        else:
            scale.append(np.nan)
    dataframe["scale"] = scale
    dataframe.to_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory containing the recaps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files or not.",
    )
    args = parser.parse_args()
    output, overwrite = Path(args.output).resolve(), args.overwrite

    if not output.exists():
        raise ValueError("Given argument 'output' is not a valid directory.")
    dataframe = pd.read_csv(output.joinpath("full_recap.csv"))
    plot_processed_recap(
        dataframe, output.joinpath("processed_recap.pdf"), overwrite
    )
    add_columns_to_recap(
        dataframe, output.joinpath("enhanced_recap.csv"), overwrite
    )
