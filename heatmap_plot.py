from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from heatmap import HeatmapResult


def build_heatmap_figure(
    heatmap: HeatmapResult,
    normalized: bool,
    background: np.ndarray | None,
    with_background: bool,
) -> Figure:
    figure = Figure(figsize=(6.2, 6.0), constrained_layout=True)
    axes = figure.subplots(1, 1, squeeze=False)
    mode_label = "Normalized" if normalized else "Raw"
    figure.suptitle(f"{mode_label} Body Volume Probability Heatmap", fontsize=14)

    axis = axes[0, 0]
    if background is not None:
        axis.imshow(
            background,
            extent=(0.0, heatmap.x_limit, heatmap.y_limit, 0.0),
            interpolation="bilinear",
        )

    display_heatmap = np.ma.masked_less_equal(heatmap.values, 0.0)
    heatmap_cmap = colormaps["turbo"].copy()
    heatmap_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    image_artist = axis.imshow(
        display_heatmap,
        extent=(0.0, heatmap.x_limit, heatmap.y_limit, 0.0),
        cmap=heatmap_cmap,
        vmin=0.0,
        vmax=float(np.nanmax(heatmap.values)),
        interpolation="bilinear",
        alpha=0.74 if with_background else 1.0,
    )
    axis.set_title("Body Volume", fontsize=10)
    axis.set_xlim(0.0, heatmap.x_limit)
    axis.set_ylim(heatmap.y_limit, 0.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    if not with_background:
        axis.set_facecolor((1.0, 1.0, 1.0, 0.0))

    colorbar = figure.colorbar(image_artist, ax=[axis], shrink=0.92, pad=0.02)
    colorbar.set_label("Probability body volume occupies location")
    if not with_background:
        figure.patch.set_alpha(0.0)
    return figure
