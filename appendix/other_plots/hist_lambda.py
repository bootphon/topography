import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        type=str,
        help="Dataframe with only the best value of lambda for each setup.",
    )
    parser.add_argument("--style")
    parser.add_argument("--use_val", action="store_true")
    args = parser.parse_args()

    dataframe = pd.read_csv(Path(args.csv).resolve(), index_col=0)
    if args.style is not None:
        plt.style.use(Path(args.style).resolve())

    lambdas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5]
    counter = Counter(dataframe.lambd.dropna())
    values = [counter[lambd] / sum(counter.values()) for lambd in lambdas]
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.bar(range(len(lambdas)), values)
    plt.xticks(range(len(lambdas)), lambdas, fontsize=15)
    plt.xlabel(r"$\lambda$", fontsize=20)
    plt.ylabel(r"Proportion", fontsize=20)
    ax.tick_params("y", labelsize=15)
    plt.savefig("histogram.pdf")
    plt.show()
