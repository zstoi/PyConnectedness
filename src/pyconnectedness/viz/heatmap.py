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
from matplotlib.animation import FuncAnimation

from ..connectedness.dynamic import DynamicConnectednessResult
from ..connectedness.static import ConnectednessResult
from ._frames import reveal_counts, window_positions, window_title


def _write_cells(texts, image, values, fmt):
    """Write the values into the cell labels, row by row"""
    rgb = image.cmap(image.norm(values))[..., :3]
    brightness = (rgb @ np.array([0.299, 0.587, 0.114])).ravel()
    for text, value, light in zip(texts, values.ravel(), brightness,
                                  strict=True):
        text.set_text("" if np.isnan(value) else fmt.format(value))
        text.set_color("white" if light < 0.5 else "black")

def spillover_heatmap(
    result,
    *,
    kind: str = "fevd",
    ax=None,
    cmap: str | None = None,
    annotate: bool = True,
    fmt: str = "{:.1f}",
    colorbar: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    animation: bool = False,
    n_frames: int = 40,
    freq: str | None = None,
    hold: int = 10,
    interval: int = 150,
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
    kind : {"fevd", "pairwise_net"}
            Matrix taken from a result, the normalized decomposition or the net
            pairwise flows. Also sets the default colormap, colour scale and
            axis labels, for a DataFrame as well.
    cmap : str
        Colormap. A diverging map such as ``"RdBu_r"`` suits net pairwise
        matrices, which are signed and centred on zero.
    annotate : bool
        Write the value into each cell.
    fmt : str
        Format string applied to the annotations.
    colorbar : bool
        Draw a colorbar next to the axes.
    animation : bool
            Return an animation instead of a still image. A rolling result is
            played back one window per frame; a static result or a matrix is
            filled in cell by cell, largest absolute value first.
        n_frames : int
        Number of windows shown, evenly spaced and ending with the last
        one. For a single matrix, the largest number of build-up steps.
    freq : str, optional
        Offset alias such as ``"QE"`` or ``"YE"``. Shows the last window of
        every period instead of ``n_frames`` windows.
    hold : int
        Number of frames the last picture is held.
    interval : int
        Delay between frames in milliseconds.
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
    if kind not in ("fevd", "pairwise_net"):
        raise ValueError(
            f"unknown kind '{kind}'; use 'fevd' or 'pairwise_net'"
        )

    rolling = isinstance(result, DynamicConnectednessResult)
    if rolling:
        cube = result.fevd if kind == "fevd" else result.pairwise_net
        cube = np.asarray(cube, dtype=float)
        matrix = pd.DataFrame(cube[-1], index=result.names,
                              columns=result.names)
    elif isinstance(result, ConnectednessResult):
        matrix = result.fevd if kind =="fevd" else result.pairwise_net
    else:
        matrix = pd.DataFrame(result)



    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")

    values = matrix.to_numpy(dtype=float)
    if animation and rolling:
        windows = window_positions(result.total.index, n_frames, freq)
        frames = cube[windows]
    else:
        frames = values[np.newaxis]

    # one colour scale for all frames; with a scale per frame the same
    # colour would stand for a different number in every window
    if "norm" not in kwargs:
        if kind == "pairwise_net":
            span = np.nanmax(np.abs(frames))
            kwargs.setdefault("vmin", -span)
            kwargs.setdefault("vmax", span)
        else:
            kwargs.setdefault("vmin", np.nanmin(frames))
            kwargs.setdefault("vmax", np.nanmax(frames))

    if kind == "pairwise_net":
        # C_ij > 0 means that i transmits more to j than it receives from j
        if cmap is None:
            cmap = "RdBu_r"
        if xlabel is None:
            xlabel = "net receiver"
        if ylabel is None:
            ylabel = "net transmitter"
    else:
        if cmap is None:
            cmap = "viridis"


    k = matrix.shape[0]
    if ax is None:
        side = 2.0 + 0.55 * k
        _, ax = plt.subplots(figsize=(side, side))

  #  values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, **kwargs)

    ticks = np.arange(k)
    ax.set_xticks(ticks)
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(xlabel if xlabel is not None else "shock source")
    ax.set_ylabel(ylabel if ylabel is not None else "affected variable")

    texts = []
    if annotate:
        # Take the text colour from the brightness of the cell rather than from
        # the value, so annotations stay readable under a diverging colormap,
        # where both ends of the scale are dark.      
        rgb = image.cmap(image.norm(values))[..., :3]
        brightness = rgb @ np.array([0.299, 0.587, 0.114])
        for (i, j), value in np.ndenumerate(values):
            shade = "white" if brightness[i, j] < 0.5 else "black"
            text = ax.text (j, i, fmt.format(value), ha="center", va="center",
                    color=shade, fontsize="small")
            texts.append(text)

    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if rolling:
        ax.set_title(window_title(result, -1))
    if not animation:
        return ax

    if rolling: 
        n_steps = len (windows)
    else:
        order = np.argsort(-np.abs(values), axis = None, kind = "stable")
        counts = reveal_counts(values.size, n_frames)
        n_steps = len(counts)
        
    def update(i):
        step = min(i, n_steps - 1)
        if rolling:
            shown = frames[step]
            ax.set_title(window_title(result, windows[step]))
        else:
            # cells not revealed yet are NaN, which imshow leaves blank
            shown = np.full(values.shape, np.nan)
            cells = order[:counts[step]]
            shown.flat[cells] = values.flat[cells]
        image.set_data(shown)
        if annotate:
            _write_cells(texts, image, shown, fmt)

        return [image]

    update (0)

    return FuncAnimation(ax.figure, update, frames = n_steps + hold, interval = interval)
