"""Script to recap the metrics and plots of the different experiments."""
import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from topography.utils import tensorboard_to_dataframe


@dataclasses.dataclass(frozen=True)
class RecapConfig:
    experiments: Path  # Path to the folder containing all the experiments.
    output: Path  # Path to the output folder, where the recaps will be.
    overwrite: bool = False  # Whether to overwrite existing files or not.


def logdir_to_recap_path(logdir: Path, start_path: str) -> str:
    return "_".join(logdir.parts[logdir.parts.index(start_path) + 1 :])


def main(config: RecapConfig) -> None:
    """Build the recaps.

    Parameters
    ----------
    config : RecapConfig
        Specified configuration.
    """
    recaps = []
    config.output.mkdir(exist_ok=True)
    start_path = config.experiments.name

    # Key function for itertools.groupby
    # Used to group plots together: plots that are to be grouped
    # have their category specified at the beginning of the file name
    # until a "_" is reached.
    keyfunc = lambda path: path.name.split("_")[0]

    for tb_root in config.experiments.rglob("tensorboard"):
        logdir = tb_root.parent
        out = config.output / logdir_to_recap_path(logdir, start_path)
        out.mkdir(exist_ok=True)
        print(str(out))

        # Make a recap of the plots
        if logdir.joinpath("plot").exists():
            img_root = sorted(logdir.joinpath("plot").glob("*.png"))
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
            with open(logdir / "environment/config.json", "r") as file:
                recap = json.load(file)
            with open(out / "config.json", "w") as file:
                json.dump(recap, file, indent=2)

            # Recover the test and val accuracy
            tb_paths = list(tb_root.glob("*"))
            if len(tb_paths) != 1:
                raise ValueError(f"More than one tb file for {logdir}.")
            df = tensorboard_to_dataframe(tb_paths[0])
            for mode in ("val", "test"):
                recap[f"{mode}_acc"] = max(
                    df[df.metric == f"{mode}/acc"].value, default=None
                )
            recaps.append(recap)
        except FileNotFoundError as error:
            print(str(error))
    # CSV containing the recap of the results
    dataframe = pd.DataFrame(recaps)
    csv_path = config.output / "full_recap.csv"
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
        required=True,
        help="Output directory containing the recaps.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files or not.",
    )

    args = parser.parse_args()
    config = RecapConfig(
        experiments=Path(args.experiments).resolve(),
        output=Path(args.output).resolve(),
        overwrite=args.overwrite,
    )
    main(config)
