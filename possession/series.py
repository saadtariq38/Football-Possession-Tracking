"""Small helpers for per-frame arrays.

A "mask" is an array of True/False, one value per frame.
"""

import numpy as np


def runs(mask):
    """Find each unbroken stretch of True in a mask.

    Returns two arrays: where each stretch starts, and the frame just after it ends.
    Example: [F, T, T, F, T] gives starts [1, 4] and stops [3, 5].
    """
    as_numbers = np.asarray(mask, dtype=np.int8)
    padded = np.concatenate(([0], as_numbers, [0]))
    # +1 where a stretch begins, -1 just after it ends.
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return starts, stops


def window_edges(n_frames, half_window):
    """For every frame, the first frame of its window and the frame just after the window.

    The window is the frame itself plus half_window frames on each side, cut off at the ends.
    """
    frame = np.arange(n_frames)
    first = np.maximum(0, frame - half_window)
    after_last = np.minimum(n_frames, frame + half_window + 1)
    return first, after_last


def rolling_mean(values, half_window):
    """For every frame, the average of the known values within half_window frames either side.

    Unknown (NaN) values are skipped. Where a window has no known value, the result is NaN.

    It uses running totals: the sum of a window is "total up to its end" minus "total up to its
    start". This gives the same answer for every frame without adding up each window again.
    """
    known = ~np.isnan(values)
    values_or_zero = np.where(known, values, 0.0)
    running_total = np.concatenate(([0.0], np.cumsum(values_or_zero)))
    running_count = np.concatenate(([0], np.cumsum(known)))

    first, after_last = window_edges(len(values), half_window)
    window_total = running_total[after_last] - running_total[first]
    window_count = running_count[after_last] - running_count[first]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = window_total / window_count
    return mean.astype(np.float32)


def rolling_any(mask, half_window):
    """For every frame, True if any frame within half_window frames either side is True."""
    running_count = np.concatenate(([0], np.cumsum(np.asarray(mask, dtype=np.int64))))
    first, after_last = window_edges(len(mask), half_window)
    window_count = running_count[after_last] - running_count[first]
    return window_count > 0
