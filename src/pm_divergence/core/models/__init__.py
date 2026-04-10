"""Core record types. All `datetime` fields MUST be timezone-aware (tzinfo set)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class MarketEvent:
    price: float
    volume: float
    liquidity: float
    timestamp: datetime
    market_id: str


@dataclass(frozen=True, slots=True)
class AttentionEvent:
    timestamp: datetime
    entity: str
    source_type: str
    engagement: float
    text: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class EventCluster:
    entity: str
    market_events: list[MarketEvent] = field(default_factory=list)
    attention_events: list[AttentionEvent] = field(default_factory=list)
