"""Cumulative returns from per-period simple returns."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cumulative_returns(returns: ArrayLike) -> NDArray[np.float64]:
    """Compounded wealth path: at index i, value is prod_{k<=i}(1 + r_k) - 1."""
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("returns must be 1-D")
    if r.size == 0:
        return r.copy()
    return np.cumprod(1.0 + r) - 1.0


def total_return(returns: ArrayLike) -> float:
    """Single scalar compounded return over the full series."""
    r = np.asarray(returns, dtype=np.float64).ravel()
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)
