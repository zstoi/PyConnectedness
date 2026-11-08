"""
Connectedness measures over time.

The rolling measures of dynamic connectedness are time series: the total
connectedness index, and the directional and net measures of every variable

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation


def plot_spillovers(
    data,
    *,
    ax=None,
    ylabel: str | None = None,
    title: str | None = None,
    zero_line: bool | None = None,
    animation: bool = False,
    sequential: bool = True,
    frames_per_line: int = 60,
    hold: int = 30,
    interval: int = 40,
    **kwargs,
):
    """
    Plot connectedness measures over time.

    Parameters
    ----------
    data : Series or DataFrame (T x k)
        One line per column, e.g. ``dyn.total`` or ``dyn.net``.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is created if omitted.
    ylabel, title : str, optional
        Axis label and title.
    zero_line : bool, optional
        Draw a horizontal line at zero. Defaults to True when the data
        change sign, as net measures do.
    animation : bool
        Return an animation in which the lines grow from left to right.
    sequential : bool
        Draw the lines of an animation one after another. If False, all
        lines grow together.
    frames_per_line : int
        Frames it takes one line to reach the end of the sample.
    hold : int
        Number of frames the finished picture is held.
    interval : int
        Delay between frames in milliseconds.
    **kwargs
        Passed to ``Axes.plot``, e.g. ``color`` or ``lw``.



    Returns
    -------
    matplotlib Axes, or FuncAnimation if ``animation`` is True

    Raises
    ------
    TypeError
        If ``data`` is neither a Series nor a DataFrame.
    ValueError
        If ``frames_per_line`` is smaller than 1.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a Series or DataFrame")
    if frames_per_line < 1:
        raise ValueError("frames_per_line must be >= 1")

    values = data.to_numpy(dtype=float)
    x = data.index.to_numpy()
    nobs, k = values.shape

    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 4.5))

    lines = []
    for j, name in enumerate(data.columns):
        line, = ax.plot(x, values[:, j], label=str(name), **kwargs)
        lines.append(line)

    # limits from the full sample; an animation would otherwise rescale the
    # axes while the lines grow
    ymin = np.nanmin(values)
    ymax = np.nanmax(values)
    pad = 0.05 * (ymax - ymin)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ymin - pad, ymax + pad)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if zero_line is None:
        zero_line = ymin < 0 < ymax
    if zero_line:
        ax.axhline(0.0, color="0.6", linewidth=0.8, zorder=0)
    if k > 1:
        ax.legend(loc="upper left", fontsize="small")

    if not animation:
        return ax

    if sequential:
        n_frames = k * frames_per_line
    else:
        n_frames = frames_per_line

    def update(i):
        started = []
        for j, line in enumerate(lines):
            start = j * frames_per_line if sequential else 0
            steps = np.clip(i + 1 - start, 0, frames_per_line)
            n = int(np.ceil(steps / frames_per_line * nobs))
            y = np.full(nobs, np.nan)
            y[:n] = values[:n, j]
            line.set_ydata(y)
            if n > 0:
                started.append(line)
        # only lines already drawn are listed; a full legend gives away
        # what comes next
        if k > 1:
            ax.legend(handles=started, loc="upper left", fontsize="small")
        return lines

    update(0)
    return FuncAnimation(ax.figure, update, frames=n_frames + hold,
                         interval=interval)
