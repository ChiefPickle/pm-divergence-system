"""
Fill simulated trades with slippage, exit by holding or signal reversal, compute PnL.

Slippage uses ``numpy.random.Generator(rng_seed)`` only — same seed ⇒ same fills (research reproducibility).
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from pm_divergence.core.models import MarketEvent
from pm_divergence.core.simulation.event_replay import SimulatedTrade

ExitReason = Literal["holding", "reverse"]


@dataclass(frozen=True, slots=True)
class TradeSimulationResult:
    trade: SimulatedTrade
    exit_time: datetime
    exit_reason: ExitReason
    entry_mid: float
    exit_mid: float
    entry_fill: float
    exit_fill: float
    """Simple long return: (exit_fill - entry_fill) / entry_fill."""
    pnl: float


def _require_aware(dt: datetime, label: str) -> None:
    if dt.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")


def _prepare_market(
    market_events: Sequence[MarketEvent],
    *,
    market_id: str | None,
) -> tuple[list[datetime], list[float]]:
    ev = [
        e
        for e in market_events
        if market_id is None or e.market_id == market_id
    ]
    ev.sort(key=lambda e: e.timestamp)
    for e in ev:
        if e.timestamp.tzinfo is None:
            raise ValueError("market event timestamps must be timezone-aware")
    return [e.timestamp for e in ev], [e.price for e in ev]


def _mid_at_or_after(ts: Sequence[datetime], prices: Sequence[float], clock: datetime) -> float:
    """First bar with timestamp >= clock (no price from strictly before clock)."""
    i = bisect_left(ts, clock)
    if i >= len(ts):
        raise ValueError("no market quote at or after requested clock")
    return float(prices[i])


def _first_reverse_time(
    entry_time: datetime,
    holding_end: datetime,
    decision_times: Sequence[datetime],
    signal_by_time: Mapping[datetime, float],
    threshold: float,
) -> datetime | None:
    """First decision clock in [entry_time, holding_end] with signal <= threshold."""
    for dt in decision_times:
        if dt < entry_time:
            continue
        if dt > holding_end:
            break
        sig = signal_by_time[dt]
        if sig <= threshold:
            return dt
    return None


def _validate_signal_coverage(decision_times: Sequence[datetime], signal_by_time: Mapping[datetime, float]) -> None:
    missing = [t for t in decision_times if t not in signal_by_time]
    if missing:
        raise KeyError(f"signal_by_time missing {len(missing)} decision time(s), e.g. {missing[0]!r}")


def simulate_trades(
    trades: Sequence[SimulatedTrade],
    market_events: Sequence[MarketEvent],
    *,
    decision_times: Sequence[datetime],
    signal_by_time: Mapping[datetime, float],
    holding_period: timedelta,
    exit_on_reverse_signal: bool = True,
    slippage_sigma: float = 1e-4,
    rng_seed: int = 0,
    market_id: str | None = None,
) -> list[TradeSimulationResult]:
    """
    Long-only simulation:

    - **Entry** mid at first ``MarketEvent.timestamp >= trade.entry_time``.
    - **Exit clock** = min(entry + holding_period, first reverse) where reverse means
      ``signal <= trade.threshold`` on ``decision_times`` (requires ``signal_by_time``).
    - **Exit** mid at first bar >= exit clock.
    - **Slippage**: i.i.d. mean-zero Gaussian noise on *relative* price,
      ``fill = mid * (1 + Normal(0, slippage_sigma))`` per leg (deterministic via ``rng_seed``).

    ``decision_times`` must be sorted strictly ascending (same grid as replay).
    """
    if holding_period < timedelta(0):
        raise ValueError("holding_period must be non-negative")
    if slippage_sigma < 0.0:
        raise ValueError("slippage_sigma must be non-negative")

    for t in decision_times:
        _require_aware(t, "decision_times")
    for a, b in zip(decision_times, decision_times[1:]):
        if b <= a:
            raise ValueError("decision_times must be strictly increasing")

    _validate_signal_coverage(decision_times, signal_by_time)

    m_ts, m_px = _prepare_market(market_events, market_id=market_id)
    if not m_ts:
        raise ValueError("no market events after optional market_id filter")

    rng = np.random.default_rng(rng_seed)
    out: list[TradeSimulationResult] = []

    for tr in trades:
        _require_aware(tr.entry_time, "trade.entry_time")
        holding_end = tr.entry_time + holding_period
        t_rev: datetime | None = None
        if exit_on_reverse_signal:
            t_rev = _first_reverse_time(
                tr.entry_time,
                holding_end,
                decision_times,
                signal_by_time,
                tr.threshold,
            )
        if t_rev is not None:
            exit_clock = min(t_rev, holding_end)
        else:
            exit_clock = holding_end

        entry_mid = _mid_at_or_after(m_ts, m_px, tr.entry_time)
        exit_mid = _mid_at_or_after(m_ts, m_px, exit_clock)

        e_noise = float(rng.normal(0.0, slippage_sigma))
        x_noise = float(rng.normal(0.0, slippage_sigma))
        entry_fill = entry_mid * (1.0 + e_noise)
        exit_fill = exit_mid * (1.0 + x_noise)

        if entry_fill <= 0.0:
            raise ValueError("non-positive entry_fill after slippage (check mid and sigma)")

        pnl = (exit_fill - entry_fill) / entry_fill
        if exit_on_reverse_signal and t_rev is not None and exit_clock == t_rev:
            reason: ExitReason = "reverse"
        else:
            reason = "holding"

        out.append(
            TradeSimulationResult(
                trade=tr,
                exit_time=exit_clock,
                exit_reason=reason,
                entry_mid=entry_mid,
                exit_mid=exit_mid,
                entry_fill=entry_fill,
                exit_fill=exit_fill,
                pnl=pnl,
            )
        )
    return out
