"""Final score from WA, lag strength, regime, and confidence (MVP confidence = 1.0)."""

from __future__ import annotations

import math

_REGIME_MULTIPLIER: dict[str, float] = {
    "information": 1.2,
    "neutral": 1.0,
    "liquidity": 0.6,
    "noise": 0.4,
}

CONFIDENCE_MVP = 1.0


def regime_multiplier(regime: str) -> float:
    """Return the scalar multiplier for a known regime label."""
    try:
        return _REGIME_MULTIPLIER[regime]
    except KeyError as e:
        raise ValueError(f"unknown regime: {regime!r}") from e


def _sigmoid(x: float) -> float:
    """Numerically stable logistic; maps ℝ → (0, 1)."""
    if not math.isfinite(x):
        return float("nan")
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def final_score(
    weighted_attention: float,
    lag_strength: float,
    regime: str,
    *,
    confidence: float | None = None,
) -> tuple[float, dict[str, float | str]]:
    """
    Score = sigmoid(WA * lag_strength * regime_multiplier * confidence).

    ``confidence`` defaults to ``1.0`` (MVP placeholder).
    Returns ``(score, raw_components)`` with ``score`` in (0, 1) for finite linear input.
    """
    conf = CONFIDENCE_MVP if confidence is None else confidence
    rm = regime_multiplier(regime)
    linear = weighted_attention * lag_strength * rm * conf
    score = _sigmoid(linear)
    raw: dict[str, float | str] = {
        "weighted_attention": weighted_attention,
        "lag_strength": lag_strength,
        "regime": regime,
        "regime_multiplier": rm,
        "confidence": conf,
        "linear_input": linear,
    }
    return score, raw
