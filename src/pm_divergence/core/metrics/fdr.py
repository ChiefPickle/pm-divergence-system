"""Benjamini–Hochberg FDR adjustment (numpy only)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def bh_qvalues(p_values: ArrayLike) -> NDArray[np.float64]:
    """
    Benjamini–Hochberg adjusted p-values (q-values), same shape/order as input.

    Values are clipped to [0, 1]. Empty input returns empty array.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size == 0:
        return p.copy()
    p = np.clip(p, 0.0, 1.0)
    m = int(p.size)
    order = np.argsort(p)
    ps = p[order]
    q_sorted = np.empty(m, dtype=np.float64)
    q_sorted[-1] = min(1.0, float(ps[-1] * m / m))
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(1.0, min(q_sorted[i + 1], float(ps[i] * m / (i + 1))))
    out = np.empty(m, dtype=np.float64)
    out[order] = q_sorted
    return out.reshape(np.asarray(p_values).shape)


def bh_rejected(p_values: ArrayLike, alpha: float) -> NDArray[np.bool_]:
    """True where BH q-value <= ``alpha`` (same shape as ``p_values``)."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    q = bh_qvalues(p_values)
    return q <= alpha
