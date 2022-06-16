"""Assign positions in 2D surfaces or 3D volumes, and compute
the inverse distance between those positions.
"""
from collections import OrderedDict
from typing import Callable, Dict

import numpy as np
import torch


def hypercube(
    num: int, dimension: int, low: float = 0, high: float = 1
) -> np.ndarray:
    """Creates `num` positions in a cube of `dimension` dimensions.
    If there exists an integer `num_axis` such that
    `num_axis**dimension == num`, the positions will be distributed
    uniformly in the cube.
    Else, the closes `num_axis` will be taken, `num_axis**dimension`
    positions are computed and only the first `num` positions are
    returned.

    Parameters
    ----------
    num : int
        Number of positions to create.
    dimension : int
        Dimension of the positions.
    low : float, optional
        Lower bound of each coordinate in the cube, by default 0.
    high : float, optional
        Higher bound of each coordinate in the cube, by default 1.

    Returns
    -------
    np.ndarray
        Array of positions, of shape (`num`, `dimension`)
    """
    num_axis = int(np.ceil(np.power(num, 1 / dimension)))
    coords = [np.linspace(low, high, num_axis) for _ in range(dimension)]
    return np.array(np.meshgrid(*coords)).reshape(dimension, -1).T[:num]


def euclidean_distance(coords: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Euclidean distance between positions.

    Parameters
    ----------
    coords : np.ndarray
        Positions of the channels, of shape (`out_channels`, `dimension`).
    eps : float, optional
        Small value in order to avoid division by zero when computing
        the inverse of this distance, by default 1e-8.

    Returns
    -------
    np.ndarray
        Matrix of distances, of shape (`out_channels`, `out_channels`).
    """
    return np.linalg.norm(coords[:, None, :] - coords, axis=-1) + eps


_DISTANCES: Dict[str, Callable] = {"euclidean": euclidean_distance}
_POSITIONS: Dict[str, Callable] = {"cube": hypercube}


def inverse_distance(
    out_channels: Dict[str, int],
    dimension: int = 2,
    norm: str = "euclidean",
    position_scheme: str = "cube",
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
        Which norm between positions to use. Must be "euclidean",
        by default "euclidean".
    position_scheme : str, optional
        How to assign positions. Must be "cube", by default "cube".

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionnary of inverse distance matrices for each Conv2d layer.
    """
    inv_dist = OrderedDict()
    assign_position = _POSITIONS[position_scheme]
    distance = _DISTANCES[norm]
    for name, channels in out_channels.items():
        coords = assign_position(channels, dimension)
        inv_dist[name] = torch.triu(
            torch.from_numpy(1 / (distance(coords) + 1)), diagonal=1
        )
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
