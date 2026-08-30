r"""
Heatmaps of variance decompositions and spillover matrices.

A connectedness result is a square matrix, and reading it as a table becomes
hard once the system has more than a handful of variables. The heatmap shows
the same numbers as colour, with rows the variable whose forecast error
variance is decomposed and columns the source of the shock.

References
----------
Diebold and Yilmaz (2012) Better to give than to receive: predictive
directional measurement of volatility spillovers. International Journal of
Forecasting, 28, 57-66.

Diebold and Yilmaz (2014) On the network topology of variance decompositions:
measuring the connectedness of financial firms. Journal of Econometrics, 182,
119-134.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..connectedness.static import ConnectednessResult


def spillover_heatmap(
    result,
    *,
    ax=None,
    cmap: str = "viridis",
    annotate: bool = True,
    fmt: str = "{:.1f}",
    colorbar: bool = True,
    **kwargs,
):
    r"""
    Plot a connectedness matrix as a heatmap.

    Parameters
    ----------
    result : ConnectednessResult or DataFrame
        A static connectedness result, whose normalized decomposition is
        used, or any square matrix with matching row and column labels. Pass
        ``result.pairwise_net`` to show net pairwise flows instead. For a
        single rolling window, wrap the stored matrix with the variable
        names, e.g.
        ``pd.DataFrame(dyn.pairwise_net[-1], index=dyn.names, columns=dyn.names)``.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is created if omitted.
    cmap : str
        Colormap. A diverging map such as ``"RdBu_r"`` suits net pairwise
        matrices, which are signed and centred on zero.
    annotate : bool
        Write the value into each cell.
    fmt : str
        Format string applied to the annotations.
    colorbar : bool
        Draw a colorbar next to the axes.
    **kwargs
        Passed to ``Axes.imshow``, e.g. ``vmin`` and ``vmax`` to centre a
        diverging colormap on zero.

    Returns
    -------
    matplotlib Axes

    Raises
    ------
    ValueError
        If the matrix is not square.
    """
    if isinstance(result, ConnectednessResult):
        matrix = result.fevd
    else:
        matrix = pd.DataFrame(result)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")

    k = matrix.shape[0]
    if ax is None:
        side = 2.0 + 0.55 * k
        _, ax = plt.subplots(figsize=(side, side))

    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, **kwargs)

    ticks = np.arange(k)
    ax.set_xticks(ticks)
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("shock to")
    ax.set_ylabel("variance of")

    if annotate:
        # Take the text colour from the brightness of the cell rather than from
        # the value, so annotations stay readable under a diverging colormap,
        # where both ends of the scale are dark.
        rgb = image.cmap(image.norm(values))[..., :3]
        brightness = rgb @ np.array([0.299, 0.587, 0.114])
        for (i, j), value in np.ndenumerate(values):
            shade = "white" if brightness[i, j] < 0.5 else "black"
            ax.text(j, i, fmt.format(value), ha="center", va="center",
                    color=shade, fontsize="small")

    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    return ax