"""Simple script to create jobs to run on CIFAR."""
import argparse
from itertools import product
from pathlib import Path

OUTPUT = "cifar"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--workdir", type=str, help="Work directory", required=True
    )
    parser.add_argument("-s", "--seed", type=int, help="Random seed", default=0)
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    jobs = Path(f"./{OUTPUT}_{args.seed}.txt").resolve()
    script = workdir / "topography/scripts/cifar/run.py"
    logdir = workdir / "experiments"
    datadir = workdir / "data"

    models = ["resnet18", "vgg16_bn", "densenet121"]
    cifar_classes = [10, 100]
    dimensions = [1, 2, 3]
    norms = ["euclidean", "l1"]
    lambdas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5]

    with open(jobs, "w", encoding="utf-8") as file:
        for model, num_classes in product(models, cifar_classes):
            path = f"cifar{num_classes}/{model}/base"
            file.write(
                f"python {script} --log {logdir / path}"
                f" --data {datadir} --seed {args.seed}"
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
                f" --data {datadir} --seed {args.seed} --model {model}"
                f" --num_classes {num_classes} --topographic --norm {norm}"
                f" --lambd {lambd} --dimension {dim}\n"
            )
