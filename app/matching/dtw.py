from __future__ import annotations

from math import inf
from typing import Sequence


def dtw_distance(left: Sequence[float], right: Sequence[float], band: int = 8) -> float:
    """Compute a small-window DTW distance for short melody contours."""
    if not left or not right:
        return inf

    n = len(left)
    m = len(right)
    band = max(band, abs(n - m))
    previous = [inf] * (m + 1)
    current = [inf] * (m + 1)
    previous[0] = 0.0

    for i in range(1, n + 1):
        start = max(1, i - band)
        end = min(m, i + band)
        current[0] = inf
        for j in range(start, end + 1):
            cost = abs(left[i - 1] - right[j - 1])
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, [inf] * (m + 1)

    return previous[m] / max(n, m)
