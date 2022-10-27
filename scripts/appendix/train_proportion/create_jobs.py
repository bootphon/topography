"""Simple script to create jobs using more or less train data."""
import argparse
from itertools import product
from pathlib import Path

OUTPUT = "traindata.txt"
LAMBD = 1
POSITION_SCHEME = "hypercube"
DIMENSION = 2
MODEL = "resnet18"
NORM = "euclidean"
SEEDS = [0, 1, 2, 3, 4]

PROPORTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--workdir", type=str, help="Work directory", required=True
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    jobs = Path(OUTPUT).resolve()
    scripts = Path(__file__).resolve().parent
    logdir = workdir / "experiments_proportion"
    datadir = workdir / "data"

    with open(jobs, "w", encoding="utf-8") as file:
        for dataset, seed, train_proportion in product(
            ["birddcase", "speechcommands"], SEEDS, PROPORTIONS
        ):
            path = f"{dataset}/{MODEL}/base"
            file.write(
                f"python {scripts}/{dataset}/run.py --log {logdir / path}"
                f" --data {datadir} --seed {seed} --model {MODEL}"
                f" --train_proportion {train_proportion}\n"
            )
            path = (
                f"{dataset}/{MODEL}/{POSITION_SCHEME}/"
                + f"dimension_{DIMENSION}/lambda_{LAMBD}/norm_{NORM}"
            )
            file.write(
                f"python {scripts}/{dataset}/run.py --log {logdir / path}"
                f" --data {datadir} --seed {seed} --model {MODEL}"
                f" --topographic --norm {NORM}"
                f" --position_scheme {POSITION_SCHEME}"
                f" --lambd {LAMBD} --dimension {DIMENSION}"
                f" --train_proportion {train_proportion}\n"
            )

        for num_classes, seed, train_proportion in product(
            [10, 100], SEEDS, PROPORTIONS
        ):
            path = f"cifar{num_classes}/{MODEL}/base"
            file.write(
                f"python {scripts}/cifar/run.py --log {logdir / path}"
                f" --data {datadir} --seed {seed} --model {MODEL}"
                f" --num_classes {num_classes}"
                f" --train_proportion {train_proportion}\n"
            )
            path = (
                f"cifar{num_classes}/{MODEL}/{POSITION_SCHEME}/"
                + f"dimension_{DIMENSION}/lambda_{LAMBD}/norm_{NORM}"
            )
            file.write(
                f"python {scripts}/cifar/run.py --log {logdir / path}"
                f" --data {datadir} --seed {seed} --model {MODEL}"
                f" --topographic --norm {NORM}"
                f" --position_scheme {POSITION_SCHEME}"
                f" --lambd {LAMBD} --dimension {DIMENSION}"
                f" --num_classes {num_classes}"
                f" --train_proportion {train_proportion}\n"
            )
