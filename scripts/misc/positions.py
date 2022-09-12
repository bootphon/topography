"""Script to plot the different position schemes used.
Cf https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-830139107
for axes limits to have the "equal" behavior in 3D plots.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from topography.core.distance import _POSITIONS


def main(num_points: int) -> None:
    for method in _POSITIONS.keys():
        for dimension in (2, 3):
            points = _POSITIONS[method](num_points, dimension)
            fig = plt.figure(figsize=(10, 10))
            ax = (
                fig.add_subplot(projection="3d")
                if dimension == 3
                else fig.add_subplot()
            )
            ax.scatter(
                *points.T,
                marker="o",
                c=np.linspace(0, 1, num_points),
                cmap="viridis",
            )
            if dimension == 3:
                box = [
                    ub - lb
                    for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")
                ]
                ax.set_box_aspect(box)
                ax.view_init(elev=19, azim=-40)

                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

                ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
                ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
                ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            else:
                ax.axis("equal")
            plt.savefig(
                f"{method}_{dimension}_{num_points}.pdf",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_points", type=int, default=256, help="Number of points."
    )
    parser.add_argument("--style", type=str, help="Path to mplstyle file.")
    args = parser.parse_args()

    if args.style is not None:
        plt.style.use(args.style)
    main(args.num_points)
