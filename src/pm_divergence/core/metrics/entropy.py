"""Empirical Shannon entropy of a signal via histogram (natural log)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def signal_entropy(x: ArrayLike, *, bins: int = 10, range_: tuple[float, float] | None = None) -> float:
    """
    Shannon entropy H = -sum p_k log(p_k) of the normalized histogram of finite values.

    Uses natural logarithm (nats). Returns NaN if there are no finite samples.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    counts, _ = np.histogram(a, bins=bins, range=range_, density=False)
    total = float(np.sum(counts))
    if total <= 0.0:
        return float("nan")
    p = counts.astype(np.float64) / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))
