"""Script to recap the metrics and plots of the different experiments."""
import argparse
import dataclasses
import itertools
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


@dataclasses.dataclass(frozen=True)
class RecapConfig:
    experiments: Path  # Path to the folder containing all the experiments.
    output: Path  # Path to the output folder, where the recaps will be.
    # Start of the experiments path, used for parsing the file names.
    start_path: str = "experiments"
    overwrite: bool = False  # Whether to overwrite existing files or not.


def process_recap(dataframe: pd.DataFrame, path: Path) -> None:
    """Process the recap dataframe. Plot the test accuracy as a function
    of lambda, for the different models and dimension of the positions.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe containing the recap of the experiments.
    path : Path
        Path to save the recap plots.
    """
    df = dataframe.dropna(axis=0)
    prod = list(product(df.model.unique(), df.num_classes.unique()))
    nrows = int(np.sqrt(len(prod)))
    ncols = len(prod) // nrows
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    for k, (model, num_classes) in enumerate(prod):
        i, j = k // nrows, k % nrows
        for dim in sorted(df.dimension.unique()):
            subdf = df[
                (df.model == model)
                & (df.num_classes == num_classes)
                & (df.dimension == dim)
            ]
            ax[i, j].scatter(
                subdf.lambd,
                subdf.test_acc,
                label=f"dim={int(dim)}",
                alpha=0.3,
                edgecolors="none",
                linewidths=2,
                s=50,
            )
        ax[i, j].legend()
        ax[i, j].set_title(f"{model}, {num_classes}")
        ax[i, j].set_xscale("log")
        ax[i, j].set_xlabel("lambda")
        ax[i, j].set_ylabel("Test acc")
    fig.savefig(path)
    plt.close()


def main(config: RecapConfig) -> None:
    """Build the recaps.

    Parameters
    ----------
    config : RecapConfig
        Specified configuration.
    """
    recaps = []
    config.output.mkdir(exist_ok=True)
    # Key function for itertools.groupby
    # Used to group plots together: plots that are to be grouped
    # have their category specified at the beginning of the file name
    # until a "_" is reached.
    keyfunc = lambda path: path.name.split("_")[0]
    for path in config.experiments.rglob("tensorboard"):
        parent = path.parent
        out = output.joinpath(  # Output directory for the current experiment
            str(parent).split(config.start_path)[1].replace("/", "_").strip("_")
        )
        out.mkdir(exist_ok=True)
        print(str(out))

        # Make a recap of the plots
        if parent.joinpath("plot").exists():
            img_root = sorted(parent.joinpath("plot").glob("*.png"))
            for key, group in itertools.groupby(img_root, keyfunc):
                if not out.joinpath(f"{key}.pdf").exists() or config.overwrite:
                    images = [Image.open(img).convert("RGB") for img in group]
                    images[0].save(
                        out.joinpath(f"{key}.pdf"),
                        save_all=True,
                        append_images=images[1:],
                    )
        try:
            # Configuration file
            with open(
                parent.joinpath("environment/config.json"),
                "r",
                encoding="utf-8",
            ) as file:
                recap = json.load(file)
            with open(
                out.joinpath("config.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(recap, file)
            # Recover the test accuracy
            test_acc = None
            with open(
                parent.joinpath("summary.log"), "r", encoding="utf-8"
            ) as file:
                lines = file.readlines()
                for line in lines:
                    if "test" in line and "acc" in line:
                        for part in line.strip().split(", "):
                            if part.startswith("acc"):
                                test_acc = float(part.removeprefix("acc"))
                        break
            with open(
                out.joinpath("summary.log"), "w", encoding="utf-8"
            ) as file:
                file.write("".join(lines))
            recap["test_acc"] = test_acc
            recaps.append(recap)
        except FileNotFoundError as error:
            print(str(error))
    # CSV containing the recap of the results
    dataframe = pd.DataFrame(recaps)
    if not output.joinpath("full_recap.csv").exists() or config.overwrite:
        dataframe.to_csv(output.joinpath("full_recap.csv"))
    process_recap(dataframe, output.joinpath("processed_recap.pdf"))


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

    parser.add_argument("--start_path", type=str, default="experiments")

    args = parser.parse_args()
    experiments = Path(args.experiments).resolve()
    output = Path(args.output).resolve()
    config = RecapConfig(
        experiments=experiments, output=output, overwrite=args.overwrite
    )
    main(config)
