"""Simple script to create jobs to plot correlation for each model."""
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
        default="./correlation.txt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files or not.",
    )
    args = parser.parse_args()

    experiments = Path(args.experiments).resolve()
    jobs = Path(args.output).resolve()
    script = Path(args.script).resolve()
    overwrite = " --overwrite" if args.overwrite else ""
    cmds = [
        f"python {script} --log {path.parent}" + overwrite
        for path in experiments.rglob("checkpoints")
        if "base" not in path.parts
    ]

    with open(jobs, "w", encoding="utf-8") as file:
        file.write("\n".join(cmds))
