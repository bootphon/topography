"""Script to plot the correlation for each layer of each model."""
import argparse
import dataclasses
from itertools import groupby
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from topography.utils.plot import tensorboard_to_dataframe

GROUP_TOGETHER = ("topo-loss",)


@dataclasses.dataclass(frozen=True)
class PlotMetricConfig:
    experiments: Path
    overwrite: bool = False
    fig_kw: Dict = dataclasses.field(
        default_factory=lambda: {"figsize": (15, 10)}
    )
    plot_kw: Dict = dataclasses.field(default_factory=dict)
    legend_kw: Dict = dataclasses.field(default_factory=dict)


def _keyfunc(name: str):
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
