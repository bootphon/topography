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
    """Creates `num_points` positions in a cube of `dimension` dimensions.
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


def _grid_in_nested(
    botleft: torch.Tensor, topright: torch.Tensor, dimension: int
) -> torch.Tensor:
    """Creates one grid to be used in `nested` in dimension `dimension`
    given the bottom left and top right corners.
    A clever loop can probably be found in order not to have to explicitely
    list the positions and to merge the cases of dimension 2 and 3.

    Parameters
    ----------
    botleft : torch.Tensor
        Bottom left vertex, of shape `dimension`.
    topright : torch.Tensor
        Top right vertex, of shape `dimension`.
    dimension : int
        Dimension of the positions.

    Returns
    -------
    torch.Tensor
        Grid to be added.

    Raises
    ------
    NotImplementedError
        If `dimension` is not 1, 2 or 3.
    """
    if dimension == 1:
        return torch.tensor([[botleft], [topright]])

    step = (topright - botleft) / 2
    if dimension == 2:
        return torch.tensor(
            [
                [botleft[0], botleft[1]],
                [botleft[0] + step[0], botleft[1]],
                [topright[0], botleft[1]],
                [botleft[0], botleft[1] + step[1]],
                [topright[0], botleft[1] + step[1]],
                [botleft[0], topright[1]],
                [botleft[0] + step[0], topright[1]],
                [topright[0], topright[1]],
            ]
        )

    if dimension == 3:
        return torch.tensor(
            [
                [botleft[0], botleft[1], botleft[2]],
                [botleft[0] + step[0], botleft[1], botleft[2]],
                [topright[0], botleft[1], botleft[2]],
                [botleft[0], botleft[1] + step[1], botleft[2]],
                [topright[0], botleft[1] + step[1], botleft[2]],
                [botleft[0], topright[1], botleft[2]],
                [botleft[0] + step[0], topright[1], botleft[2]],
                [topright[0], topright[1], botleft[2]],
                [botleft[0], botleft[1], botleft[2] + step[2]],
                [topright[0], botleft[1], botleft[2] + step[2]],
                [botleft[0], topright[1], botleft[2] + step[2]],
                [topright[0], topright[1], botleft[2] + step[2]],
                [botleft[0], botleft[1], topright[2]],
                [botleft[0] + step[0], botleft[1], topright[2]],
                [topright[0], botleft[1], topright[2]],
                [botleft[0], botleft[1] + step[1], topright[2]],
                [topright[0], botleft[1] + step[1], topright[2]],
                [botleft[0], topright[1], topright[2]],
                [botleft[0] + step[0], topright[1], topright[2]],
                [topright[0], topright[1], topright[2]],
            ]
        )
    raise NotImplementedError(f"No scale grid for dimension {dimension} > 3.")


def nested(
    num_points: int,
    dimension: int,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    nested_factor: float = 0.5,
) -> torch.Tensor:
    """Creates `num_points` positions from nested regular grids
    in dimension `dimension`.
    Each grid has one position at each of its vertices (2**dimension)
    and one in the center of each of its edges (dimension*2**(dimension-1)).

    Parameters
    ----------
    num_points : int
        Number of positions to create.
    dimension : int
        Dimension of the positions.
    lower_bound : float, optional
        Lower bound of each coordinate, by default 0.0.
    upper_bound : float, optional
        Upper bound of each coordinate, by default 1.0.
    nested_factor : float, optional
        Factor to scale down each nested grid by.
        For a given grid with sides of length L,
        the next one inside will have all
        its sides of length `nested_factor`*L. By default 0.5.

    Returns
    -------
    torch.Tensor
        Tensor of positions, of shape (`num_points`, `dimension`).
    """
    if dimension > 1:
        points_per_step = 2**dimension + dimension * 2 ** (dimension - 1)
    else:  # In 1D: 2 points are added each step, not 3
        points_per_step = 2
    steps = num_points // points_per_step
    if num_points % points_per_step:
        steps += 1
    points = []
    botleft = lower_bound * torch.ones(dimension)
    topright = upper_bound * torch.ones(dimension)
    for _ in range(steps):
        points.append(_grid_in_nested(botleft, topright, dimension))
        step = (1 - nested_factor) * (topright - botleft) / 2
        botleft += step
        topright -= step
    return torch.vstack(points)[:num_points]


def hypersphere(
    num_points: int, dimension: int, scale_factor: float = 0.5
) -> torch.Tensor:
    """Creates `num_points` regular positions on a 1-sphere (a circle)
    or a 2-sphere (the usual sphere).
    The Fibonacci sphere algorithm is used for the usual sphere, with
    the implementation from https://stackoverflow.com/a/26127012
    following [fibonacci]_.

    Parameters
    ----------
    num_points : int
        Number of positions to create.
    dimension : int
        Dimension of the positions. Must be either 2 (for a circle) or 3
        (for a usual sphere).
    scale_factor : float, optional
        Scale factor: with scale_factor == 1, the hypersphere has coordinates
        between -1 and 1. By default 0.5, in order for the coords to be between
        -0.5 and 0.5.

    Returns
    -------
    torch.Tensor
        Tensor of regular positions on the hypersphere,
        of shape (`num_points`, `dimension`).

    Raises
    ------
    NotImplementedError
        If the dimension is not 2 or 3.

    References
    ----------
    .. [fibonacci] González, Á. (2010). Measurement of areas on a
    sphere using Fibonacci and latitude–longitude lattices.
    Mathematical Geosciences, 42(1), 49-64.
    """
    if dimension == 2:  # 1-sphere: circle
        time = torch.arange(0, num_points)
        omega = 2 * math.pi / num_points
        positions = [torch.cos(omega * time), torch.sin(omega * time)]
        return scale_factor * torch.vstack(positions).T

    if dimension == 3:  # 2-sphere: usual sphere, Fibonacci algorithm
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians
        time = torch.arange(0, num_points)
        ycoord = 1 - (time / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius = torch.sqrt(1 - ycoord * ycoord)  # radius at y
        theta = phi * time  # golden angle increment
        positions = [
            torch.cos(theta) * radius,
            ycoord,
            torch.sin(theta) * radius,
        ]
        return scale_factor * torch.vstack(positions).T

    raise NotImplementedError(f"No sphere for dim {dimension}, only 2 and 3.")


_DISTANCES: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "euclidean": lambda coords: torch.cdist(coords, coords, p=2),
    "l1": lambda coords: torch.cdist(coords, coords, p=1),
}
_POSITIONS: Dict[str, Callable[..., torch.Tensor]] = {
    "hypercube": hypercube,
    "nested": nested,
    "sphere": hypersphere,
}


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
        How to assign positions. Must be "hypercube", "nested" or
        "hypersphere", by default "hypercube".

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
    import numpy as np

    square_x, square_y = hypercube(64, 2).T
    plt.scatter(square_x, square_y)
    plt.show()

    cube_x, cube_y, cube_z = hypercube(256, 3).T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(cube_x, cube_y, cube_z, marker="o")
    plt.show()

    plt.figure(figsize=(15, 10))
    NUM_POINTS, FACTOR = 64, 0.7
    nested_grid = nested(NUM_POINTS, 2, nested_factor=FACTOR)
    cmap = plt.get_cmap("viridis", lut=NUM_POINTS // 8)
    for start in range(0, NUM_POINTS, 8):
        plt.scatter(
            *nested_grid[start : start + 8].T,
            s=10,
            color=cmap.colors[start // 8],
        )
    plt.axis("equal")
    plt.show()

    nested_cube = nested(256, 3)
    assert nested_cube.shape == torch.Size([256, 3])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*nested_cube.T, marker="o")·
    plt.show()

    NUM_POINTS = 7
    int_square = hypercube(NUM_POINTS, 2, integer_positions=True)
    plt.scatter(int_square[:, 0], int_square[:, 1])
    plt.title(f"Scatter integer positions: {NUM_POINTS} points in 2d")
    plt.show()

    naxis = int_square.max().item() + 1
    img = np.ma.array(
        np.arange(naxis * naxis),
        mask=np.arange(naxis * naxis) >= len(int_square),
    ).reshape(naxis, naxis)
    plt.imshow(img, origin="lower")
    plt.title(f"Imshow using {NUM_POINTS} points")
    plt.show()

    fig, ax = plt.subplots(nrows=naxis, ncols=naxis)
    for k, (i, j) in enumerate(int_square):
        ax[naxis - 1 - j, i].imshow(np.ones((5, 5)))
        ax[naxis - 1 - j, i].set_title(f"{i}, {j}")

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].axis("off")
    plt.suptitle(f"Subplots with {NUM_POINTS} points")
    plt.show()
