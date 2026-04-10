"""
Event-driven replay: score from data with timestamp <= t only; entries at t + latency.

Research rule: ``compute_score`` must not use information outside the provided snapshot
(no hidden globals / future rows). This module does not enforce that — reviewers should.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from pm_divergence.core.models import AttentionEvent, MarketEvent


@dataclass(frozen=True, slots=True)
class EventReplaySnapshot:
    """Causal view at ``asof``: only events whose ``timestamp <= asof`` (inclusive)."""

    asof: datetime
    market_events: tuple[MarketEvent, ...]
    attention_events: tuple[AttentionEvent, ...]


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    decision_time: datetime
    signal: float
    entry_time: datetime
    threshold: float


def snapshot_at(
    asof: datetime,
    market_events: Sequence[MarketEvent],
    attention_events: Sequence[AttentionEvent],
) -> EventReplaySnapshot:
    """Materialize the strict <= asof filter (explicit, testable).

    Events are ordered by ``timestamp`` ascending, matching :func:`event_replay`.
    """
    if asof.tzinfo is None:
        raise ValueError("asof must be timezone-aware")
    m = sorted((e for e in market_events if e.timestamp <= asof), key=lambda e: e.timestamp)
    a = sorted((e for e in attention_events if e.timestamp <= asof), key=lambda e: e.timestamp)
    return EventReplaySnapshot(asof=asof, market_events=tuple(m), attention_events=tuple(a))


def _require_aware(ts: Sequence[datetime], label: str) -> None:
    if any(t.tzinfo is None for t in ts):
        raise ValueError(f"{label} must be timezone-aware")


def _require_strictly_increasing(ts: Sequence[datetime]) -> None:
    for a, b in zip(ts, ts[1:]):
        if b <= a:
            raise ValueError("decision_times must be strictly increasing")


def event_replay(
    decision_times: Sequence[datetime],
    market_events: Sequence[MarketEvent],
    attention_events: Sequence[AttentionEvent],
    compute_score: Callable[[EventReplaySnapshot], float],
    *,
    threshold: float,
    latency: timedelta,
) -> list[SimulatedTrade]:
    """
    For each decision time ``t`` (ascending):

    1. Build ``snapshot`` = all market/attention events with ``timestamp <= t``.
    2. ``signal = compute_score(snapshot)`` (must not read outside ``snapshot``).
    3. If ``signal > threshold``, append a trade with ``entry_time = t + latency``.

    Pre-sorts inputs by event time so each step uses time-ordered prefix slices (no future rows).
    """
    if latency < timedelta(0):
        raise ValueError("latency must be non-negative")
    _require_aware(decision_times, "decision_times")
    _require_strictly_increasing(decision_times)

    m_sorted = sorted(market_events, key=lambda e: e.timestamp)
    a_sorted = sorted(attention_events, key=lambda e: e.timestamp)
    for e in m_sorted + a_sorted:
        if e.timestamp.tzinfo is None:
            raise ValueError("all event timestamps must be timezone-aware")

    m_ts = [e.timestamp for e in m_sorted]
    a_ts = [e.timestamp for e in a_sorted]

    trades: list[SimulatedTrade] = []
    for t in decision_times:
        i_m = bisect_right(m_ts, t)
        i_a = bisect_right(a_ts, t)
        snapshot = EventReplaySnapshot(
            asof=t,
            market_events=tuple(m_sorted[:i_m]),
            attention_events=tuple(a_sorted[:i_a]),
        )
        signal = compute_score(snapshot)
        if signal > threshold:
            trades.append(
                SimulatedTrade(
                    decision_time=t,
                    signal=signal,
                    entry_time=t + latency,
                    threshold=threshold,
                )
            )
    return trades
