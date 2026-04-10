"""
Synthetic dataset + report builder (shared by CLI, API, dashboard).

Deterministic given ``SystemConfig``, dataset ``seed``, and slippage ``rng_seed``.
No ML/RL — plain numpy + explicit causal snapshots only.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
import numpy as np

from pm_divergence.backtest.engine import SystemConfig
from pm_divergence.core.metrics.fdr import bh_qvalues
from pm_divergence.core.metrics.pnl import total_return
from pm_divergence.core.metrics.sharpe import sharpe_ratio
from pm_divergence.core.models import AttentionEvent, MarketEvent
from pm_divergence.core.scoring.attention_score import (
    semantic_relevance_score,
    source_weighting,
    weighted_attention,
)
from pm_divergence.core.scoring.final_score import final_score
from pm_divergence.core.scoring.lag_model import best_lag
from pm_divergence.core.scoring.regime_detector import classify_regime, regime_score
from pm_divergence.core.simulation.event_replay import EventReplaySnapshot, event_replay, snapshot_at
from pm_divergence.core.simulation.trade_simulator import simulate_trades

_REF_EMB = [1.0, 0.0]


def to_np_dt64(dt: datetime) -> np.datetime64:
    if dt.tzinfo is None:
        return np.datetime64(dt, "ns")
    naive_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(naive_utc, "ns")


def load_synthetic_dataset(
    *,
    hours: int = 96,
    seed: int = 42,
) -> tuple[list[MarketEvent], list[AttentionEvent], list[datetime]]:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    decision_times = [base + timedelta(hours=i) for i in range(hours)]

    market: list[MarketEvent] = []
    price = 0.5
    for i, t in enumerate(decision_times):
        price = float(np.clip(price + rng.normal(0.0, 0.012), 0.05, 0.95))
        market.append(
            MarketEvent(
                price=price,
                volume=float(800 + rng.integers(-80, 80)),
                liquidity=4000.0,
                timestamp=t,
                market_id="synthetic",
            )
        )

    attention: list[AttentionEvent] = []
    for i in range(0, hours, 5):
        attention.append(
            AttentionEvent(
                timestamp=decision_times[i] + timedelta(minutes=20),
                entity="syn",
                source_type="tweet",
                engagement=float(5 + rng.integers(0, 40)),
                text="",
                embedding=[1.0, 0.0],
            )
        )
    return market, attention, decision_times


def score_snapshot(snap: EventReplaySnapshot, *, regime_enabled: bool) -> float:
    terms: list[tuple[float, float, float]] = []
    for a in snap.attention_events:
        emb = list(a.embedding)
        if len(emb) < 2:
            emb = emb + [0.0] * (2 - len(emb))
        terms.append(
            (
                a.engagement,
                source_weighting(a.source_type),
                semantic_relevance_score(emb[:2], _REF_EMB),
            )
        )
    wa = weighted_attention(terms) if terms else 0.0

    m = snap.market_events
    if len(m) < 8:
        return final_score(wa, 0.0, "neutral")[0]

    px = np.array([x.price for x in m], dtype=np.float64)
    reg = regime_from_snapshot(snap, regime_enabled=regime_enabled)

    ts = np.array([to_np_dt64(x.timestamp) for x in m])
    wa_series = np.array(
        [
            float(
                sum(
                    e.engagement
                    for e in snap.attention_events
                    if e.timestamp <= row.timestamp
                )
            )
            for row in m
        ],
        dtype=np.float64,
    )
    lag_res = best_lag(ts, wa_series, px, min_pairs=6)
    lag_s = (
        0.0
        if lag_res.best_lag is None or not math.isfinite(lag_res.lag_strength)
        else lag_res.lag_strength
    )
    return final_score(wa, lag_s, reg)[0]


def regime_from_snapshot(snap: EventReplaySnapshot, *, regime_enabled: bool) -> str:
    m = snap.market_events
    if len(m) < 8 or not regime_enabled:
        return "neutral"
    px = np.array([x.price for x in m], dtype=np.float64)
    volumes = np.array([x.volume for x in m], dtype=np.float64)
    vol = np.abs(np.diff(px, prepend=px[0]))
    rs = regime_score(volumes, vol)
    reg_labels = classify_regime(rs)
    return str(reg_labels.reshape(-1)[-1])


def global_best_lag(
    market: list[MarketEvent],
    attention: list[AttentionEvent],
) -> tuple[str | None, float]:
    m = sorted(market, key=lambda e: e.timestamp)
    if len(m) < 8:
        return None, float("nan")
    ts = np.array([to_np_dt64(x.timestamp) for x in m])
    px = np.array([x.price for x in m], dtype=np.float64)
    wa_series = np.array(
        [
            float(
                sum(e.engagement for e in attention if e.timestamp <= row.timestamp)
            )
            for row in m
        ],
        dtype=np.float64,
    )
    res = best_lag(ts, wa_series, px, min_pairs=8)
    if res.best_lag is None:
        return None, float("nan")
    return str(res.best_lag), res.lag_strength


def sanitize_for_json(x: object) -> object:
    if isinstance(x, dict):
        return {str(k): sanitize_for_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize_for_json(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def build_synthetic_report(
    cfg: SystemConfig,
    *,
    seed: int = 42,
    rng_seed: int = 7,
) -> dict[str, object]:
    if str(cfg.slippage_model).lower() != "gaussian":
        raise ValueError(f"Unsupported slippage_model {cfg.slippage_model!r}")

    market, attention, decision_times = load_synthetic_dataset(seed=seed)

    signal_by_time: dict[datetime, float] = {}
    for t in decision_times:
        signal_by_time[t] = score_snapshot(
            snapshot_at(t, market, attention),
            regime_enabled=cfg.regime_enabled,
        )

    trades = event_replay(
        decision_times,
        market,
        attention,
        lambda snap: signal_by_time[snap.asof],
        threshold=cfg.threshold,
        latency=timedelta(seconds=cfg.latency_seconds),
    )

    sim = simulate_trades(
        trades,
        market,
        decision_times=decision_times,
        signal_by_time=signal_by_time,
        holding_period=timedelta(hours=cfg.holding_period_hours),
        slippage_sigma=cfg.slippage_sigma,
        rng_seed=rng_seed,
        market_id="synthetic",
    )

    pnls = np.array([r.pnl for r in sim], dtype=np.float64)
    total_pnl = float(total_return(pnls)) if pnls.size else 0.0
    hit_rate = float(np.mean(pnls > 0.0)) if pnls.size else 0.0
    sharpe = sharpe_ratio(pnls, periods_per_year=252.0) if pnls.size >= 2 else float("nan")

    p_vals = np.array([1.0 - signal_by_time[t] for t in decision_times], dtype=np.float64)
    p_vals = np.clip(p_vals, 1e-12, 1.0)
    fdr = float(np.mean(bh_qvalues(p_vals)))

    lag_label, lag_strength = global_best_lag(market, attention)

    return {
        "total_pnl": total_pnl,
        "hit_rate": hit_rate,
        "sharpe": sharpe,
        "fdr": fdr,
        "best_lag": {
            "delta": lag_label,
            "lag_strength": lag_strength,
        },
    }


def build_dashboard_series(
    cfg: SystemConfig,
    *,
    seed: int = 42,
    rng_seed: int = 7,
) -> dict[str, object]:
    """Series for UI: timeline (signal + regime), cumulative PnL steps, hit rate."""
    if str(cfg.slippage_model).lower() != "gaussian":
        raise ValueError(f"Unsupported slippage_model {cfg.slippage_model!r}")

    market, attention, decision_times = load_synthetic_dataset(seed=seed)

    signal_by_time: dict[datetime, float] = {}
    for t in decision_times:
        snap = snapshot_at(t, market, attention)
        signal_by_time[t] = score_snapshot(snap, regime_enabled=cfg.regime_enabled)

    trades = event_replay(
        decision_times,
        market,
        attention,
        lambda snap: signal_by_time[snap.asof],
        threshold=cfg.threshold,
        latency=timedelta(seconds=cfg.latency_seconds),
    )

    sim = simulate_trades(
        trades,
        market,
        decision_times=decision_times,
        signal_by_time=signal_by_time,
        holding_period=timedelta(hours=cfg.holding_period_hours),
        slippage_sigma=cfg.slippage_sigma,
        rng_seed=rng_seed,
        market_id="synthetic",
    )

    timeline: list[dict[str, object]] = []
    for t in decision_times:
        snap = snapshot_at(t, market, attention)
        timeline.append(
            {
                "time": t,
                "signal": signal_by_time[t],
                "regime": regime_from_snapshot(snap, regime_enabled=cfg.regime_enabled),
            }
        )

    pnl_steps: list[tuple[datetime, float]] = []
    running = 1.0
    for r in sorted(sim, key=lambda x: x.exit_time):
        running *= 1.0 + r.pnl
        pnl_steps.append((r.exit_time, running - 1.0))
    if not pnl_steps and decision_times:
        pnl_steps.append((decision_times[0], 0.0))

    hit_rate = float(np.mean(np.array([s.pnl > 0.0 for s in sim], dtype=np.float64))) if sim else 0.0

    return {
        "timeline": timeline,
        "pnl_steps": pnl_steps,
        "hit_rate": hit_rate,
        "n_trades": len(sim),
    }
