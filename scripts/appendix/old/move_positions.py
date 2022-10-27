import numpy as np
import torch


def move_positions(
    positions: torch.Tensor,
    seed: int,
    pairwise_distance: float,
    fwhm_prop_distance: float = 0.5,
) -> torch.Tensor:
    fwhm = fwhm_prop_distance * pairwise_distance
    # sigma = 1 / 4 * 1 / 7 * 1 / 2.355
    sigma = fwhm / 2.355
    num_points, dimension = positions.shape
    generator = np.random.default_rng(seed=seed)
    normal = torch.tensor(
        generator.multivariate_normal(
            np.zeros(dimension), sigma * np.eye(dimension), num_points
        )
    )
    return positions + normal
