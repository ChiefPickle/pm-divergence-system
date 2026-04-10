"""Process-local in-memory stores (MVP)."""

from __future__ import annotations

from dataclasses import dataclass, field

from pm_divergence.core.models import AttentionEvent, MarketEvent


@dataclass
class AppState:
    attention: list[AttentionEvent] = field(default_factory=list)
    market: list[MarketEvent] = field(default_factory=list)
    reports: dict[str, dict] = field(default_factory=dict)


store = AppState()


def get_store() -> AppState:
    return store
