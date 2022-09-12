"""Script to plot the metrics logged in TensorBoard for a given experiment.
The topographic losses are grouped in the same figure.
"""
import argparse
import dataclasses
from itertools import groupby
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from topography.utils import tensorboard_to_dataframe

GROUP_TOGETHER = ("topo-loss",)


@dataclasses.dataclass(frozen=True)
class PlotMetricConfig:
    logdir: Path  # Output directory.
    overwrite: bool = False  # Whether to overwrite exisiting files or not
    fig_kw: Dict = dataclasses.field(
        default_factory=lambda: {"figsize": (15, 10)}
    )  # Figure specifications
    plot_kw: Dict = dataclasses.field(default_factory=dict)  # Plot config
    legend_kw: Dict = dataclasses.field(default_factory=dict)  # Legend config


def _keyfunc(name: str) -> str:
    """Function to generate the key according to which the metrics will
    be grouped by with itertools.groupby.

    Parameters
    ----------
    name : str
        Name of the metric: train/loss, test/acc,
        'train/extras/topo-loss/layer1.0.conv1'...

    Returns
    -------
    str
        If the name of the metric does not have a part in `GROUP_TOGETHER`,
        the key is the name itself (this group will only have one element).
        Else, the key is the beginning of the name until the group
        is found.
        For example, the key returned for "test/acc" is "test/acc"
        while the one returned for "train/extras/topo-loss/features.0" is
        "train/extras/topo-loss", which is also the one returned
        for "train/extras/topo-loss/features.40".
    """
    for group in GROUP_TOGETHER:
        if group in name:
            return name.split(group)[0] + group
    return name


def main(config: PlotMetricConfig):
    logdir = config.logdir
    plotdir = logdir / "plot"
    plotdir.mkdir(exist_ok=True)

    for idx, experiment_id in enumerate((logdir / "tensorboard").glob("*")):
        dataframe = tensorboard_to_dataframe(str(experiment_id))
        all_metrics = dataframe.metric.unique()
        for metric, group in groupby(all_metrics, _keyfunc):
            name = plotdir.joinpath(
                f"metrics_{metric.replace('/', '-')}-{idx}.pdf"
            )
            if not name.exists() or config.overwrite:
                to_save = False
                plt.figure(**config.fig_kw)
                for label in group:
                    subdf = dataframe[dataframe.metric == label]
                    if len(subdf.value) > 1:
                        to_save = True
                    plt.plot(
                        subdf.step,
                        subdf.value,
                        label=label,
                        **config.plot_kw,
                    )
                plt.legend(**config.legend_kw)
                plt.xlabel("Step")
                plt.ylabel(metric)
                if to_save:
                    plt.savefig(name)
                    plt.savefig(name.with_suffix(".png"))
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files or not.",
    )
    args = parser.parse_args()
    config = PlotMetricConfig(
        logdir=Path(args.log).resolve(),
        overwrite=args.overwrite,
        legend_kw={"fontsize": "x-small"},
    )
    main(config)
