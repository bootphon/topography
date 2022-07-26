"""Script to plot the metrics logged in TensorBoard for each experiment.
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
    experiments: Path  # Path to the directory containing all the experiments
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
        Else,
        For example,
    """
    for group in GROUP_TOGETHER:
        if group in name:
            return name.split(group)[0] + group
    return name


def main(config: PlotMetricConfig):
    for path in config.experiments.rglob("tensorboard"):
        parent = path.parent
        print(parent)
        output = parent.joinpath("plot")
        output.mkdir(exist_ok=True)

        for idx, experiment_id in enumerate(path.glob("*")):
            dataframe = tensorboard_to_dataframe(str(experiment_id))
            all_metrics = dataframe.metric.unique()
            for metric, group in groupby(all_metrics, _keyfunc):
                name = output.joinpath(
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
        "-x",
        "--experiments",
        type=str,
        required=True,
        help="Experiments directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files or not.",
    )
    args = parser.parse_args()
    config = PlotMetricConfig(
        experiments=Path(args.experiments).resolve(),
        overwrite=args.overwrite,
        legend_kw={"fontsize": "x-small"},
    )
    main(config)
