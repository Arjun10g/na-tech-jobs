"""Confidence + provenance types shared across the cascade tiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class Extraction:
    """One extractor's verdict on a single feature.

    ``confidence`` is heuristic on Tier 1 (regex): 1.0 for strict patterns,
    0.7-0.9 for loose ones. On Tier 2 (LLM) it's the model's own probability
    when available, otherwise a flat 0.6 default. ``None`` everywhere when the
    extractor saw nothing relevant.
    """

    value: Any
    confidence: float
    source: str  # "regex" | "llm" | "structured"
    rule_id: str | None = None

    def as_meta(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "rule_id": self.rule_id,
        }


# Below this threshold a Tier-1 result triggers a Tier-2 escalation.
TIER1_THRESHOLD: float = 0.6
