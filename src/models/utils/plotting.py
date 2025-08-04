from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy import ndarray

from src.data.utils import load_image


def _make_matching_figure(
    image0: ndarray,
    image1: ndarray,
    points0: ndarray,
    points1: ndarray,
    colors: ndarray,
    enable_line: bool = True,
    dpi: int = 75,
    pad: float = 1.0,
    text: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    for ax, image in zip(axes, [image0, image1]):
        ax.imshow(image)
        ax.axis("off")
    plt.tight_layout(pad=pad)

    if len(points0) != 0 and len(points0) == len(points1):
        fig.canvas.draw()
        if enable_line:
            fig_points0 = axes[0].transData.transform(points0)
            fig_points1 = axes[1].transData.transform(points1)
            fig_points0 = fig.transFigure.inverted().transform(fig_points0)
            fig_points1 = fig.transFigure.inverted().transform(fig_points1)
            for i in range(len(points0)):
                x = fig_points0[i, 0], fig_points1[i, 0]
                y = fig_points0[i, 1], fig_points1[i, 1]
                line = Line2D(
                    x, y, c=colors[i], lw=2, transform=fig.transFigure
                )
                fig.add_artist(line)

        axes[0].autoscale(enable=False)
        axes[1].autoscale(enable=False)
        axes[0].scatter(points0[:, 0], points0[:, 1], c=colors[:, :3], s=4)
        axes[1].scatter(points1[:, 0], points1[:, 1], c=colors[:, :3], s=4)

    if text is not None:
        text = "\n".join(text)
        color = "k" if image0[:100, :200].mean() > 200 else "w"
        fig.text(
            0.01,
            0.99,
            text,
            c=color,
            va="top",
            ha="left",
            fontsize=15,
            transform=axes[0].transAxes,
        )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return None
    else:
        return fig


def _make_colormap(
    errors: ndarray, threshold: float, alpha: float = 1.0
) -> ndarray:
    x = 1.0 - (errors / (threshold * 2.0)).clip(min=0.0, max=1.0)
    colormap = np.stack(
        [2.0 - x * 2.0, x * 2.0, np.zeros_like(x), np.ones_like(x) * alpha],
        axis=-1,
    ).clip(min=0.0, max=1.0)
    return colormap


def make_matching_figure(
    path0: str,
    path1: str,
    points0: ndarray,
    points1: ndarray,
    errors: ndarray,
    threshold: float,
    text: Optional[List[str]] = None,
    **kwargs,
) -> Optional[Figure]:
    image0, _, _ = load_image(path0, mode="color")
    image1, _, _ = load_image(path1, mode="color")
    colors = _make_colormap(errors, threshold, alpha=0.1)
    text = [f"#matches: {len(points0)}"] if text is None else text
    figure = _make_matching_figure(
        image0, image1, points0, points1, colors, text=text, **kwargs
    )
    return figure
