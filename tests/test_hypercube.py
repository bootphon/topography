"""Test of the hypercube function that is used to to assign positions
to the channels.
"""
import math
from typing import Dict, Tuple

import pytest
import torch

from topography.core.distance import hypercube

expected_grids: Dict[Tuple[int, int], Dict[bool, torch.Tensor]] = {
    (5, 1): {
        True: torch.tensor([[0], [1], [2], [3], [4]]),
        False: torch.tensor([[0], [0.25], [0.5], [0.75], [1.0]]),
    },
    (9, 2): {
        True: torch.tensor(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ]
        ),
        False: torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.5],
                [0.0, 1.0],
                [0.5, 1.0],
                [1.0, 1.0],
            ]
        ),
    },
    (7, 2): {
        True: torch.tensor(
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2]]
        ),
        False: torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.5],
                [0.0, 1.0],
            ]
        ),
    },
}


@pytest.mark.parametrize(
    "num_points,dimension,integer_positions",
    [
        (5, 1, True),
        (5, 1, False),
        (9, 2, True),
        (9, 2, False),
        (7, 2, True),
        (7, 2, False),
    ],
)
def test_simple_grid(num_points, dimension, integer_positions):
    grid = hypercube(num_points, dimension, integer_positions=integer_positions)
    expected_grid = expected_grids[(num_points, dimension)][integer_positions]
    assert torch.equal(grid, expected_grid)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_hypercube(dimension):
    num_axis = 10
    num_points = num_axis**dimension
    coords = hypercube(num_points, dimension)
    assert coords.shape == (num_points, dimension)

    other_num_points = num_axis**dimension - num_axis + 1
    other_coords = hypercube(other_num_points, dimension)
    assert torch.equal(coords[:other_num_points], other_coords)


@pytest.mark.parametrize("num_points", [2, 11, 20, 64, 256, 512, 1000, 111111])
def test_explicit_expression(num_points):
    dimension = 2
    num_axis = int(math.ceil(num_points ** (1 / dimension)))

    ref = hypercube(num_points, dimension)
    ref_integer = hypercube(num_points, dimension, integer_positions=True)
    assert torch.allclose(ref, ref_integer / (num_axis - 1))

    explicit = torch.zeros((num_points, dimension))
    for idx in range(num_points):
        explicit[idx][0] = idx % num_axis
        explicit[idx][1] = idx // num_axis
    assert torch.allclose(ref, explicit / (num_axis - 1))
