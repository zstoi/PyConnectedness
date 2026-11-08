"""
Frame selection for the animated figures.

An animation either plays back a rolling result, one estimation window per
frame, or builds up a single picture step by step
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def window_positions(index, n_frames, freq=None):
    """
    Positions of the rolling windows shown in an animation.

    Parameters
    ----------
    index : Index
        Index of the rolling result, one entry per window.
    n_frames : int
        Number of windows, evenly spaced over the sample.
    freq : str, optional
        Offset alias such as "QE" or "YE". Takes the last window of every
        period instead and overrides ``n_frames``.

    Returns
    -------
    ndarray of int
        Increasing positions that always end with the last window.

    Raises
    ------
    ValueError
        If ``n_frames`` is smaller than 1, or ``freq`` is given for an index
        without dates.
    """
    if freq is not None:
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("freq requires a result with a DatetimeIndex")
        # the last window of a period is the value an end-of-quarter or
        # end-of-year report would quote
        position = pd.Series(np.arange(len(index)), index=index)
        last = position.resample(freq).last()
        return last.dropna().to_numpy(dtype=int)

    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    steps = np.linspace(0, len(index) - 1, min(n_frames, len(index)))
    return np.unique(steps.round().astype(int))


def window_title(result, t):
    """Window end date and total connectedness of window t."""
    date = result.total.index[t]
    if isinstance(date, pd.Timestamp):
        date = date.strftime("%Y-%m-%d")
    total = result.total.iloc[t]
    return f"window ending {date}, total connectedness {total:.1f}%"


def reveal_counts(n_items, n_frames):
    """
    Number of items visible after each step of a build-up.

    Parameters
    ----------
    n_items : int
        Items to reveal, e.g. the cells of a matrix or the edges of a graph.
    n_frames : int
        Largest number of steps.

    Returns
    -------
    ndarray of int
        One count per step, increasing, the last equal to ``n_items``. Empty
        if there is nothing to reveal.

    Raises
    ------
    ValueError
        If ``n_frames`` is smaller than 1.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if n_items == 0:
        return np.array([], dtype=int)
    # several items per step when there are more items than frames
    n_steps = min(n_frames, n_items)
    return np.ceil(np.arange(1, n_steps + 1) * n_items / n_steps).astype(int)
