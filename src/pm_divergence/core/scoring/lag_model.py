"""Lag correlation: WA(t) vs forward price at first bar >= t + Δ (no time-axis lookahead)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

LAG_DELTAS: tuple[np.timedelta64, ...] = (
    np.timedelta64(5, "m"),
    np.timedelta64(15, "m"),
    np.timedelta64(1, "h"),
    np.timedelta64(6, "h"),
)

_DEFAULT_MIN_PAIRS = 10
_STD_EPS = 1e-15


@dataclass(frozen=True, slots=True)
class LagModelResult:
    best_lag: np.timedelta64 | None
    lag_strength: float


def _delta_ns(delta: np.timedelta64) -> int:
    return int(np.asarray(delta, dtype="timedelta64[ns]").astype(np.int64))


def _pearson(x: NDArray[np.float64], y: NDArray[np.float64], *, min_pairs: int) -> float:
    if x.size < min_pairs:
        return float("nan")
    if np.std(x, ddof=0) <= _STD_EPS or np.std(y, ddof=0) <= _STD_EPS:
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else float("nan")


def _forward_pairs(
    timestamps: NDArray[np.datetime64],
    wa: NDArray[np.float64],
    price: NDArray[np.float64],
    delta: np.timedelta64,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Align WA at t with price at the first timestamp >= t + delta (simulated forward shift)."""
    targets = timestamps + np.asarray(delta, dtype="timedelta64[ns]")
    j = np.searchsorted(timestamps, targets, side="left")
    valid = j < timestamps.size
    return wa[valid], price[j[valid]]


def lag_correlation_for_delta(
    timestamps: ArrayLike,
    wa: ArrayLike,
    price: ArrayLike,
    delta: np.timedelta64,
    *,
    min_pairs: int = _DEFAULT_MIN_PAIRS,
) -> float:
    """Pearson correlation between WA(t) and price at first bar >= t + delta; nan if insufficient."""
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    w = np.asarray(wa, dtype=np.float64)
    p = np.asarray(price, dtype=np.float64)
    if ts.shape != w.shape or ts.shape != p.shape:
        raise ValueError("timestamps, wa, and price must have the same shape")
    if ts.size == 0:
        return float("nan")
    if not np.all(np.diff(ts.astype(np.int64)) > 0):
        raise ValueError("timestamps must be strictly increasing")
    x, y = _forward_pairs(ts, w, p, delta)
    return _pearson(x, y, min_pairs=min_pairs)


def best_lag(
    timestamps: ArrayLike,
    wa: ArrayLike,
    price: ArrayLike,
    *,
    min_pairs: int = _DEFAULT_MIN_PAIRS,
    deltas: tuple[np.timedelta64, ...] = LAG_DELTAS,
) -> LagModelResult:
    """Pick Δ in ``deltas`` with largest |corr|; ties break toward shorter Δ."""
    scored: list[tuple[np.timedelta64, float]] = []
    for d in deltas:
        r = lag_correlation_for_delta(timestamps, wa, price, d, min_pairs=min_pairs)
        if np.isfinite(r):
            scored.append((d, r))
    if not scored:
        return LagModelResult(best_lag=None, lag_strength=float("nan"))
    best_delta, best_r = max(scored, key=lambda dr: (abs(dr[1]), -_delta_ns(dr[0])))
    return LagModelResult(best_lag=best_delta, lag_strength=best_r)
