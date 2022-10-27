"""Script to process the recap made with `run.py`: new plots and new columns
in the csv.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topography.utils import tensorboard_to_dataframe


def plot_processed_recap(
    dataframe: pd.DataFrame, path: Path, overwrite: bool
) -> None:
    """Process the recap dataframe. Plot the best val accuracy as a function
    of lambda, for each dimension, for each pair of model and dataset; and
    for every norm used.

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
    datasets_models_pairs = [
        (dataset, model)
        for model in dataframe.model.unique()
        for dataset in dataframe.dataset.unique()
    ]
    nrows = len(dataframe.dataset.unique())
    ncols = len(dataframe.model.unique())

    for norm in dataframe.norm.dropna().unique():
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(15, 15),
            sharex="row",
            sharey="row",
        )
        for pair_index, (dataset, model) in enumerate(datasets_models_pairs):
            row, column = pair_index % nrows, pair_index // nrows

            df_model = dataframe.query(
                f"(model == '{model}') & (dataset == '{dataset}') & topographic"
            )
            reference = dataframe.query(
                f"(model == '{model}') & (dataset == '{dataset}')"
                f" & ~topographic"
            ).val_acc.mean()

            if df_model.empty:
                continue
            if not np.isnan(reference):
                ax[row, column].hlines(
                    reference,
                    df_model.lambd.min(),
                    df_model.lambd.max(),
                    label="ref",
                    colors="k",
                    zorder=1,
                )

            for dim in sorted(df_model.dimension.unique()):
                df_dim_norm = df_model[
                    (df_model.dimension == dim) & (df_model.norm == norm)
                ]
                ax[row, column].scatter(
                    df_dim_norm.lambd,
                    df_dim_norm.val_acc,
                    label=f"dim={int(dim)}",
                    alpha=0.3,
                    edgecolors="none",
                    linewidths=2,
                    s=50,
                    zorder=2,
                )
            ax[row, column].legend(loc="lower left")
            ax[row, column].set_title(f"{model}, {dataset}")
            ax[row, column].set_xscale("log")
            ax[row, column].set_xlabel("lambda")
            ax[row, column].set_ylabel("Best val acc")
        plt.suptitle(f"Norm {norm}", fontsize=20)
        plt.tight_layout()
        fig.savefig(path.parent / (path.stem + f"_{norm}" + path.suffix))
        plt.close()


def add_columns_to_recap(
    dataframe: pd.DataFrame, path: Path, overwrite: bool
) -> None:
    """Adds columns to the recap CSV. For now, only adds the weight of
    the topographic loss compared to the cross entropy.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe containing the recap of the experiments.
    path : Path
        Path to save the recap with the new columns.
    overwrite : bool
        Whether to overwrite existing files or not.

    Raises
    ------
    ValueError
        If the path to the TensorBoard file is invalid.
    """
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
    dataframe = pd.read_csv(output.joinpath("full_recap.csv"), index_col=0)
    plot_processed_recap(
        dataframe, output.joinpath("processed_recap.pdf"), overwrite
    )
    add_columns_to_recap(
        dataframe, output.joinpath("enhanced_recap.csv"), overwrite
    )