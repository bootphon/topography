"""Simple script to create jobs to run on CIFAR."""
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
        default="./cifar.txt",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    jobs = Path(args.output).resolve()
    script = workdir / "topography/scripts/cifar/run.py"
    logdir = workdir / "experiments"
    datadir = workdir / "data"

    models = ["resnet18", "vgg16_bn", "densenet121"]
    cifar_classes = [10, 100]
    dimensions = [1, 2, 3]
    norms = ["euclidean", "l1"]
    lambdas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]

    with open(jobs, "w", encoding="utf-8") as file:
        for model, num_classes in product(models, cifar_classes):
            path = f"cifar{num_classes}/{model}/base"
            file.write(
                f"python {script} --log {logdir / path}"
                f" --data {datadir} --seed {SEED}"
                f" --model {model} --num_classes {num_classes}\n"
            )
        for model, num_classes, dim, norm, lambd in product(
            models, cifar_classes, dimensions, norms, lambdas
        ):
            path = (
                f"cifar{num_classes}/{model}/"
                + f"dimension_{dim}/lambda_{lambd}/norm_{norm}"
            )
            file.write(
                f"python {script} --log {logdir / path}"
                f" --data {datadir} --seed {SEED} --model {model}"
                f" --num_classes {num_classes} --topographic --norm {norm}"
                f" --lambd {lambd} --dimension {dim}\n"
            )
