from __future__ import annotations

import math
from collections.abc import Iterable

_SOURCE_WEIGHTS: dict[str, float] = {
    "tier1_news": 1.5,
    "expert": 1.2,
    "tweet": 1.0,
    "viral": 0.5,
    "bot": 0.2,
}


def source_weighting(source_type: str) -> float:
    """Return the fixed scalar weight for a known `source_type`."""
    try:
        return _SOURCE_WEIGHTS[source_type]
    except KeyError as e:
        raise ValueError(f"unknown source_type: {source_type!r}") from e


def semantic_relevance_score(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two equal-length embedding vectors (no external ML)."""
    if len(vec_a) != len(vec_b):
        raise ValueError("embedding lengths must match")
    if not vec_a:
        return 0.0
    dot = math.fsum(x * y for x, y in zip(vec_a, vec_b, strict=True))
    na = math.sqrt(math.fsum(x * x for x in vec_a))
    nb = math.sqrt(math.fsum(y * y for y in vec_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def weighted_attention(terms: Iterable[tuple[float, float, float]]) -> float:
    """WA = sum(attention * weight * srs) over each (attention, weight, srs) term."""
    return math.fsum(att * wt * srs for att, wt, srs in terms)
