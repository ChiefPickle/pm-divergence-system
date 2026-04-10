"""Simplified Sharpe: mean excess / std, scaled by sqrt(periods_per_year)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def sharpe_ratio(
    returns: ArrayLike,
    *,
    periods_per_year: float = 1.0,
    risk_free: float = 0.0,
) -> float:
    """
    (mean(r - rf) / std(r)) * sqrt(periods_per_year).

    Population std (ddof=0). Returns NaN if ``len(r) < 2`` or std is degenerate.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    if r.size < 2:
        return float("nan")
    x = r - risk_free
    s = float(np.std(x, ddof=0))
    if s <= 1e-15 or not np.isfinite(s):
        return float("nan")
    m = float(np.mean(x))
    if not np.isfinite(m):
        return float("nan")
    return (m / s) * float(np.sqrt(periods_per_year))
