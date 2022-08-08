"""Simple script to create jobs to run on SpeechCommands."""
import argparse
from itertools import product
from pathlib import Path

SEED: int = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--workdir", type=str, help="Work directory", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output containing the list of jobs to run.",
        default="./speechcommands.txt",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    jobs = Path(args.output).resolve()
    script = workdir / "topography/scripts/speechcommands/run.py"
    logdir = workdir / "experiments"
    datadir = workdir / "data"

    models = ["resnet18", "vgg16_bn", "densenet121"]
    dimensions = [1, 2, 3]
    norms = ["euclidean", "l1"]
    lambdas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]

    with open(jobs, "w", encoding="utf-8") as file:
        for model in models:
            path = f"speechcommands/{model}/base"
            file.write(
                f"python {script} --log {logdir / path}"
                f" --data {datadir} --seed {SEED} --model {model}\n"
            )
        for model, dim, norm, lambd in product(
            models, dimensions, norms, lambdas
        ):
            path = (
                f"speechcommands/{model}/"
                + f"dimension_{dim}/lambda_{lambd}/norm_{norm}"
            )
            file.write(
                f"python {script} --log {logdir / path}"
                f" --data {datadir} --seed {SEED} --model {model}"
                f" --topographic --norm {norm}"
                f" --lambd {lambd} --dimension {dim}\n"
            )
