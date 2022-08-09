"""Script to recap the metrics and plots of the different experiments."""
import argparse
import dataclasses
import itertools
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


@dataclasses.dataclass(frozen=True)
class RecapConfig:
    experiments: Path  # Path to the folder containing all the experiments.
    output: Path  # Path to the output folder, where the recaps will be.
    # Start of the experiments path, used for parsing the file names.
    start_path: str = "experiments"
    overwrite: bool = False  # Whether to overwrite existing files or not.


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
        out = config.output.joinpath(  # Directory for the current experiment
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
            # Recover the test and val accuracy
            accuracies = defaultdict(list)
            with open(
                parent.joinpath("summary.log"), "r", encoding="utf-8"
            ) as file:
                lines = file.readlines()
                for line in lines:
                    for mode in ("val", "test"):
                        if mode in line and "acc" in line:
                            for part in line.strip().split(", "):
                                if part.startswith("acc"):
                                    accuracies[mode].append(
                                        float(part.removeprefix("acc"))
                                    )
            with open(
                out.joinpath("summary.log"), "w", encoding="utf-8"
            ) as file:
                file.write("".join(lines))
            for mode in ("val", "test"):
                recap[f"{mode}_acc"] = max(accuracies[mode])
            recaps.append(recap)
        except FileNotFoundError as error:
            print(str(error))
    # CSV containing the recap of the results
    dataframe = pd.DataFrame(recaps)
    csv_path = config.output.joinpath("full_recap.csv")
    if not csv_path.exists() or config.overwrite:
        dataframe.to_csv(csv_path)


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
