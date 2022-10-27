"""Simple script to create all jobs to run."""
import argparse
from itertools import product
from pathlib import Path

DATASETS = ["speechcommands", "birddcase", "cifar10", "cifar100"]
SEEDS = [0, 1, 2, 3, 4]

MODELS = ["resnet18", "vgg16_bn", "densenet121"]
DIMENSIONS = [1, 2, 3]
NORMS = ["euclidean", "l1"]
POSITION_SCHEMES = ["hypercube", "nested", "hypersphere"]
LAMBDAS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", type=str, help="Experiments directory")
    parser.add_argument("data", type=str, help="Data directory")
    args = parser.parse_args()

    expdir = Path(args.experiments).resolve()
    datadir = Path(args.data).resolve()
    scriptdir = Path(__file__).resolve().parent

    jobs = []
    for dataset, seed in product(DATASETS, SEEDS):
        if dataset.startswith("cifar"):
            dataset, num_classes = "cifar", int(dataset.removeprefix("cifar"))
        script = scriptdir / f"{dataset}.py"
        for model in MODELS:
            path = f"{dataset}/{model}/base"
            job = (
                f"python {script} --log {expdir / path}"
                f" --data {datadir} --seed {seed} --model {model}"
            )
            if dataset == "cifar":
                job += f" --num_classes {num_classes}"
            jobs.append(job)
        for model, dim, norm, lambd, position_scheme in product(
            MODELS, DIMENSIONS, NORMS, LAMBDAS, POSITION_SCHEMES
        ):
            if position_scheme == "hypersphere" and dim == 1:
                continue  # No sphere in dimension 1
            path = (
                f"{dataset}/{model}/{position_scheme}/"
                + f"dimension_{dim}/lambda_{lambd}/norm_{norm}"
            )
            job = (
                f"python {script} --log {expdir / path}"
                f" --data {datadir} --seed {seed} --model {model}"
                f" --topographic --norm {norm}"
                f" --position_scheme {position_scheme}"
                f" --lambd {lambd} --dimension {dim}"
            )
            if dataset == "cifar":
                job += f" --num_classes {num_classes}"
            jobs.append(job)
    Path("./jobs.txt").write_text("\n".join(jobs))
