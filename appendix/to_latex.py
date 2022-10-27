import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

GROUPBY = [
    "model",
    "topographic",
    "lambd",
    "dimension",
    "norm",
    "position_scheme",
    "dataset",
]

INDICES = {
    (2.0, "hypercube"): "2D grid",
    (2.0, "hypersphere"): "Circle",
    (2.0, "nested"): "2D nested",
    (3.0, "hypercube"): "3D grid",
    (3.0, "hypersphere"): "Sphere",
    (3.0, "nested"): "3D nested",
}

MODELS = (
    ("DenseNet121", "ResNet18", "VGG16"),
    ("densenet121", "resnet18", "vgg16_bn"),
)
DATASETS = (
    (("CIFAR10", "CIFAR100"), ("cifar10", "cifar100")),
    (("SpeechCommands", "BirdDCASE"), ("speechcommands", "birddcase")),
)


def aggregate_seeds_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = defaultdict(list)
    for values, subdfs in dataframe.groupby(GROUPBY, dropna=False):
        for cat, val in zip(GROUPBY, values):
            output[cat].append(val)
        acc_df = subdfs.groupby("seed").mean()
        for metric in ["test_acc", "val_acc"]:
            output[f"{metric}_mean"].append(acc_df[metric].mean() * 100)
            output[f"{metric}_std"].append(acc_df[metric].std() * 100)
        output["seeds"].append(len(acc_df))
        output["log"].append(subdfs[subdfs.seed == 0].log.iloc[0])
    return pd.DataFrame.from_dict(output)


def format_before_latex(results, datasets):
    out = pd.concat(
        {
            formatted_dataset: pd.concat(
                {
                    formatted_name: results[
                        (results.model == model) & (results.dataset == dataset)
                    ][["dimension", "position_scheme", "acc"]]
                    .groupby(["dimension", "position_scheme"], dropna=False)
                    .sum()
                    for formatted_name, model in zip(*MODELS)
                },
                axis=1,
            )
            for formatted_dataset, dataset in zip(*datasets)
        },
        axis=1,
    ).droplevel(level=2, axis=1)
    out.set_index(
        pd.Index((INDICES.get(idx, "Baseline") for idx in out.index), name=""),
        inplace=True,
    )
    return out


def main(dataframe: pd.DataFrame, outpath: Path, precision: int) -> None:
    if outpath.exists() and not outpath.is_dir():
        raise ValueError(f"Wrong path {output}")
    outpath.mkdir(exist_ok=True)

    def format(row):
        return (
            f"${row[f'test_acc_mean']:0.{precision}f} \pm"
            + f" {row[f'test_acc_std']:0.{precision}f}$"
        )

    output = aggregate_seeds_df(dataframe)
    output.to_csv(outpath / "seed_averaged.csv")

    idx = []
    for _, y in output[output.topographic == True].groupby(
        ["model", "dimension", "norm", "position_scheme", "dataset"]
    ):
        idx.append(y["val_acc_mean"].idxmax())

    results = output.loc[
        (output.topographic == False) | (output.index.isin(idx))
    ].copy()
    results["acc"] = results.apply(format, axis=1)
    results.to_csv(outpath / "best.csv")

    final = []
    for datasets in DATASETS:
        good = format_before_latex(results, datasets)
        style = good.style.highlight_max(axis="index", props="bfseries: ;")
        final.append(
            style.to_latex(
                multicol_align="c",
                position="ht",
                hrules=True,
            )
        )

    patched = "\n".join(final).replace("\\bfseries $", "$ \mathbf{")
    correct = re.sub(r"(\{[0-9]*[.][0-9]+(?!\})\b)", r"\1}", patched)
    (outpath / "results.tex").write_text(correct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to the csv file."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./", help="Output file."
    )
    parser.add_argument(
        "--precision", type=int, default=2, help="Float precision."
    )
    args = parser.parse_args()

    dataframe = pd.read_csv(Path(args.csv).resolve(), index_col=0)
    main(dataframe, Path(args.output).resolve(), args.precision)
