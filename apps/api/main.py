"""FastAPI MVP: ingest, signal, synthetic backtest, report by id."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pm_divergence.backtest.engine import load_system_config
from pm_divergence.backtest.synthetic import build_synthetic_report, sanitize_for_json, score_snapshot
from pm_divergence.core.models import AttentionEvent, MarketEvent
from pm_divergence.core.simulation.event_replay import snapshot_at

from apps.api.state import AppState, get_store, store

app = FastAPI(title="pm-divergence-system API", version="0.1.0")


class AttentionIn(BaseModel):
    timestamp: datetime
    entity: str
    source_type: str
    engagement: float
    text: str = ""
    embedding: list[float] = Field(default_factory=list)


class MarketIn(BaseModel):
    price: float
    volume: float
    liquidity: float
    timestamp: datetime
    market_id: str


class BacktestRunIn(BaseModel):
    seed: int = 42
    rng_seed: int = 7


@app.post("/ingest/attention")
def ingest_attention(body: AttentionIn, st: AppState = Depends(get_store)) -> dict[str, str]:
    ev = AttentionEvent(
        timestamp=body.timestamp,
        entity=body.entity,
        source_type=body.source_type,
        engagement=body.engagement,
        text=body.text,
        embedding=list(body.embedding),
    )
    st.attention.append(ev)
    return {"status": "ok"}


@app.post("/ingest/market")
def ingest_market(body: MarketIn, st: AppState = Depends(get_store)) -> dict[str, str]:
    ev = MarketEvent(
        price=body.price,
        volume=body.volume,
        liquidity=body.liquidity,
        timestamp=body.timestamp,
        market_id=body.market_id,
    )
    st.market.append(ev)
    return {"status": "ok"}


@app.get("/signal/{entity}")
def get_signal(entity: str, st: AppState = Depends(get_store)) -> dict:
    cfg = load_system_config()
    m = [e for e in st.market if e.market_id == entity]
    a = [e for e in st.attention if e.entity == entity]
    times = [e.timestamp for e in m] + [e.timestamp for e in a]
    asof = max(times) if times else datetime.now(timezone.utc)
    snap = snapshot_at(asof, m, a)
    sig = score_snapshot(snap, regime_enabled=cfg.regime_enabled)
    return {
        "entity": entity,
        "asof": asof.isoformat(),
        "signal": sig,
        "market_events": len(m),
        "attention_events": len(a),
    }


@app.post("/backtest/run")
def run_backtest(body: BacktestRunIn = BacktestRunIn()) -> dict[str, str]:
    cfg = load_system_config()
    if str(cfg.slippage_model).lower() != "gaussian":
        raise HTTPException(400, "only gaussian slippage is supported")
    report = build_synthetic_report(cfg, seed=body.seed, rng_seed=body.rng_seed)
    rid = str(uuid.uuid4())
    store.reports[rid] = sanitize_for_json(report)  # type: ignore[assignment]
    return {"id": rid}


@app.get("/report/{report_id}")
def get_report(report_id: str, st: AppState = Depends(get_store)) -> dict:
    if report_id not in st.reports:
        raise HTTPException(404, "report not found")
    out = dict(st.reports[report_id])
    out["id"] = report_id
    return out
