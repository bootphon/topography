"""Simple script to create jobs to run on SpeechCommands."""
import argparse
from itertools import product
from pathlib import Path

OUTPUT = "speechcommands"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--workdir", type=str, help="Work directory", required=True
    )
    parser.add_argument("-s", "--seed", type=int, help="Random seed", default=0)
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    jobs = Path(f"./{OUTPUT}_{args.seed}.txt").resolve()
    script = Path(__file__).resolve().parent / "run.py"
    logdir = workdir / "experiments"
    datadir = workdir / "data"

    models = ["resnet18", "vgg16_bn", "densenet121"]
    dimensions = [1, 2, 3]
    norms = ["euclidean", "l1"]
    position_schemes = ["hypercube", "nested", "hypersphere"]
    lambdas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5]

    with open(jobs, "w", encoding="utf-8") as file:
        for model in models:
            path = f"speechcommands/{model}/base"
            file.write(
                f"python {script} --log {logdir / path}"
                f" --data {datadir} --seed {args.seed} --model {model}\n"
            )
        for model, dim, norm, lambd, position_scheme in product(
            models, dimensions, norms, lambdas, position_schemes
        ):
            if position_scheme == "hypersphere" and dim == 1:
                continue  # No sphere in dimension 1
            path = (
                f"speechcommands/{model}/{position_scheme}/"
                + f"dimension_{dim}/lambda_{lambd}/norm_{norm}"
            )
            file.write(
                f"python {script} --log {logdir / path}"
                f" --data {datadir} --seed {args.seed} --model {model}"
                f" --topographic --norm {norm}"
                f" --position_scheme {position_scheme}"
                f" --lambd {lambd} --dimension {dim}\n"
            )
