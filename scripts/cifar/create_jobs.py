"""Simple script to create jobs to run on CIFAR."""
import argparse
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
    script = workdir.joinpath("topography/scripts/cifar/run.py")
    logdir = workdir.joinpath("experiments")
    datadir = workdir.joinpath("data")

    with open(jobs, "w", encoding="utf-8") as file:
        for model in ["alexnet", "resnet18", "vgg16_bn"]:
            for num_classes in [10]:
                path = f"cifar{num_classes}/{model}/base"
                file.write(
                    f"python {script} --log {logdir.joinpath(path)}"
                    f" --data {datadir} --seed {SEED}"
                    f" --model {model} --num_classes {num_classes}\n"
                )
        for model in ["alexnet", "resnet18", "vgg16_bn"]:
            for num_classes in [10]:
                for dimension in [1, 2, 3]:
                    for lambd in [0.1, 0.5, 1, 10]:
                        path = (
                            f"cifar{num_classes}/{model}/"
                            + f"dimension_{dimension}/lambda_{lambd}"
                        )
                        file.write(
                            f"python {script} --log {logdir.joinpath(path)}"
                            f" --data {datadir} --seed {SEED} --model {model}"
                            f" --num_classes {num_classes} --topographic"
                            f" --lambd {lambd} --dimension {dimension}\n"
                        )
