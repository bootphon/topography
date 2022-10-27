"""Check if two models have the same state_dict
https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212
"""
import argparse
from collections import OrderedDict

import torch


def validate_state_dicts(
    first_dict: OrderedDict, second_dict: OrderedDict
) -> bool:
    if len(first_dict) != len(second_dict):
        print(f"Length mismatch: {len(first_dict)}, {len(second_dict)}")
        return False

    # Replicate modules have "module" attached to their keys,
    # so strip these off when comparing to local model.
    if next(iter(first_dict.keys())).startswith("module"):
        first_dict = {k[len("module") + 1 :]: v for k, v in first_dict.items()}

    if next(iter(second_dict.keys())).startswith("module"):
        second_dict = {
            k[len("module") + 1 :]: v for k, v in second_dict.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        first_dict.items(), second_dict.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs=2, help="Path to state dicts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_path, second_path = args.models
    print(
        validate_state_dicts(
            torch.load(first_path, map_location=device),
            torch.load(second_path, map_location=device),
        )
    )
