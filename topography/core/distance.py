"""Assign positions in 2D surfaces or 3D volumes, and compute
the inverse distance between those positions.
"""
import math
from typing import Callable, Dict

import torch


def hypercube(
    num_points: int,
    dimension: int,
    lower_bound: float = 0,
    upper_bound: float = 1,
    integer_positions: bool = False,
) -> torch.Tensor:
    """Creates `num` positions in a cube of `dimension` dimensions.
    If there exists an integer `num_axis` such that
    `num_axis**dimension == num_points`, the positions will be distributed
    uniformly in the cube.
    Else, the closes `num_axis` will be taken, `num_axis**dimension`
    positions are computed and only the first `num` positions are
    returned.

    Parameters
    ----------
    num_points : int
        Number of positions to create.
    dimension : int
        Dimension of the positions. 1D for a line, 2D for a square grid,
        3D for a grid in a cube, etc.
    lower_bound : float, optional
        Lower bound of each coordinate in the cube, by default 0.
    upper_bound : float, optional
        Higher bound of each coordinate in the cube, by default 1.
    integer_positions: bool, optional
        Whether to use integer positions instead of specifying
        the lower and upper bounds of each coordinate. If True,
        `lower_bound` and `upper_bound` are ignored, by default False.

    Returns
    -------
    torch.Tensor
        Tensor of positions, of shape (`num_points`, `dimension`).
    """
    num_axis = int(math.ceil(num_points ** (1 / dimension)))
    if integer_positions:
        coords = [torch.arange(0, num_axis) for _ in range(dimension)]
    else:
        coords = [
            torch.linspace(lower_bound, upper_bound, num_axis)
            for _ in range(dimension)
        ]
    return (
        torch.cat(torch.meshgrid(*coords, indexing="xy"))
        .reshape(dimension, -1)
        .T[:num_points]
    )


_DISTANCES: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "euclidean": lambda coords: torch.cdist(coords, coords, p=2),
    "l1": lambda coords: torch.cdist(coords, coords, p=1),
}
_POSITIONS: Dict[str, Callable[..., torch.Tensor]] = {"hypercube": hypercube}


def inverse_distance(
    out_channels: Dict[str, int],
    dimension: int = 2,
    norm: str = "euclidean",
    position_scheme: str = "hypercube",
) -> Dict[str, torch.Tensor]:
    """Compute the inverse distance matrices to be used in the topographic
    loss. Function called when creating the TopographicModel.
    Each matrix is upper triangular in order to remove
    redundant distances.

    Parameters
    ----------
    out_channels : Dict[str, int]
        Dictionnary containing the number of output channels
        of each Conv2d layer.
    dimension : int, optional
        Dimension of the position assigned to each channel
        of each Conv2d layer, by default 2.
    norm : str, optional
        Which norm between positions to use. Must be "euclidean" or "l1",
        by default "euclidean".
    position_scheme : str, optional
        How to assign positions. Must be "hypercube", by default "hypercube".

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionnary of inverse distance matrices for each Conv2d layer.
    """
    inv_dist = {}
    assign_position = _POSITIONS[position_scheme]
    distance = _DISTANCES[norm]
    for layer, num_channels in out_channels.items():
        coords = assign_position(num_channels, dimension)
        inv_dist[layer] = torch.triu(1 / (distance(coords) + 1), diagonal=1)
    return inv_dist


if __name__ == "__main__":  # pragma: no cover
    # Simple check in order to visualize how the positions are distributed.
    import matplotlib.pyplot as plt

    square_x, square_y = hypercube(64, 2).T
    plt.scatter(square_x, square_y)
    plt.show()

    cube_x, cube_y, cube_z = hypercube(256, 3).T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(cube_x, cube_y, cube_z, marker="o")
    plt.show()
