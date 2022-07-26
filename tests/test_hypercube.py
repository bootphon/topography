"""Test of the hypercube function that is used to to assign positions
to the channels.
"""
import pytest
import torch

from topography.core.distance import hypercube


@pytest.mark.parametrize("integer_positions", [True, False])
def test_simple_grid(integer_positions):
    grid = hypercube(9, 2, integer_positions=integer_positions)
    expected_grid = (
        torch.tensor(
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
        )
        if integer_positions
        else torch.tensor(
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
        )
    )
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
