"""Regime labels from volume/volatility z-score ratio (fixed thresholds, no I/O)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Denominator stabilization (matches spec).
_EPS_DENOM = 1e-6
# Treat population std below this as degenerate (constant series).
_STD_FLOOR = 1e-15

# Fixed score → label mapping (deterministic, documented cutpoints).
_TH_INFORMATION = 1.0
_TH_LIQUIDITY = -1.0
_TH_NEUTRAL_ABS = 0.35


def _zscore(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Population z-score; constant input yields zeros (no NaNs from div-by-zero)."""
    if x.size == 0:
        return x.copy()
    m = np.mean(x)
    s = np.std(x, ddof=0)
    if s <= _STD_FLOOR:
        return np.zeros_like(x, dtype=np.float64)
    return (x - m) / s


def regime_score(volume: ArrayLike, volatility: ArrayLike) -> NDArray[np.float64]:
    """regime_score = zscore(volume) / (zscore(volatility) + 1e-6)."""
    v = np.asarray(volume, dtype=np.float64)
    w = np.asarray(volatility, dtype=np.float64)
    if v.shape != w.shape:
        raise ValueError("volume and volatility must have the same shape")
    z_v = _zscore(v)
    z_w = _zscore(w)
    return z_v / (z_w + _EPS_DENOM)


def classify_regime(regime_scores: ArrayLike) -> NDArray[np.str_]:
    """Map each regime score to one of: information | neutral | liquidity | noise."""
    s = np.asarray(regime_scores, dtype=np.float64)
    flat = s.ravel()
    nan_m = ~np.isfinite(flat)
    c_information = flat >= _TH_INFORMATION
    c_liquidity = flat <= _TH_LIQUIDITY
    c_neutral = np.abs(flat) <= _TH_NEUTRAL_ABS
    # np.select uses the first matching condition; non-finite scores → neutral.
    out = np.select(
        [nan_m, c_information, c_liquidity, c_neutral],
        ["neutral", "information", "liquidity", "neutral"],
        default="noise",
    )
    return out.reshape(s.shape)
