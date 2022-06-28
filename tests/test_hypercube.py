"""Test of the hypercube function that is used to to assign positions
to the channels.
"""
import pytest
import torch

from topography.core.distance import hypercube


def test_simple_grid():
    grid = hypercube(9, 2)
    expected_grid = torch.Tensor(
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
