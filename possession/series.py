"""Small helpers for per-frame arrays: runs of True, and centred rolling windows.

All windows are centred and half-open the way the rest of the repository treats intervals, and all
are computed with cumulative sums, so the cost does not grow with the window.
"""

import numpy as np


def runs(mask):
    """(starts, stops) of the maximal True runs, half-open."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _window_bounds(n, half_window):
    index = np.arange(n)
    return np.maximum(0, index - half_window), np.minimum(n, index + half_window + 1)


def rolling_mean(values, half_window):
    """Mean of the non-NaN values within +/-half_window frames; NaN where there are none."""
    known = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(known, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(known)))
    low, high = _window_bounds(len(values), half_window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((total[high] - total[low]) / (count[high] - count[low])).astype(np.float32)


def rolling_any(mask, half_window):
    """True where any frame within +/-half_window frames is True."""
    count = np.concatenate(([0], np.cumsum(np.asarray(mask, dtype=np.int64))))
    low, high = _window_bounds(len(mask), half_window)
    return (count[high] - count[low]) > 0
