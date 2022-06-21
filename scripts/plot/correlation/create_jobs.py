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
    args = parser.parse_args()

    experiments = Path(args.experiments).resolve()
    jobs = Path(args.output).resolve()
    script = Path(args.script).resolve()
    cmds = [
        f"python {script} --log {path.parent}"
        for path in experiments.rglob("*")
        if path.name == "checkpoints" and "base" not in path.parts
    ]

    with open(jobs, "w", encoding="utf-8") as file:
        file.write("\n".join(cmds))
