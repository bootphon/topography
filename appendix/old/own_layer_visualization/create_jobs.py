"""Simple script to create jobs to plot the map of maximum activating images of
each channel of each layer for each model.
"""
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x",
        "--experiments",
        type=str,
        help="Experiments directory",
        required=True,
    )
    parser.add_argument(
        "-s", "--script", type=str, help="Path to script", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output containing the list of jobs to run.",
        default="./layers.txt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files or not.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=100,
        help="Number of optimizer iterations.",
    )
    args = parser.parse_args()

    experiments = Path(args.experiments).resolve()
    jobs = Path(args.output).resolve()
    script = Path(args.script).resolve()
    overwrite = " --overwrite" if args.overwrite else ""

    cmds = []
    for path in experiments.rglob("checkpoints"):
        if "base" in path.parts:
            continue
        dimension_ok = False
        for part in path.parts:
            if part.startswith("dimension_") and int(
                part.removeprefix("dimension_")
            ) in (1, 2):
                dimension_ok = True
                break
        if dimension_ok:
            cmds.append(
                f"python {script} --log {path.parent} --n_iter {args.n_iter}"
                + overwrite
            )

    with open(jobs, "w", encoding="utf-8") as file:
        file.write("\n".join(cmds))
