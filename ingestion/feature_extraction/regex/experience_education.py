"""Years of experience + minimum education extractors."""

from __future__ import annotations

import re

from ingestion.feature_extraction.confidence import Extraction

# "5+ years", "5-7 years", "minimum 5 years", "at least 5 years"
YEARS_RE = re.compile(
    r"\b(?:(?:at\s+least|min(?:imum)?|over)\s+)?(\d{1,2})\s*(?:\+|-\s?(\d{1,2}))?\s*"
    r"(?:years?|yrs?)\b\s*(?:of)?\s*(?:experience|exp\.?|in\s+the\s+field)?",
    re.IGNORECASE,
)

# Education tiers, ordered cheapest → most demanding. Each tier's pattern
# requires *degree-context* to fire (e.g. plain "MA" and "MS" without
# "Master's" or "degree" nearby is too ambiguous — they could be state codes,
# abbreviations like "Microsoft", etc.).
EDU_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "phd",
        re.compile(
            r"\b(?:PhD|Ph\.?\s?D\.?|Doctorate|Doctoral\s+degree)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "masters",
        re.compile(
            r"\b(?:Master[''']?s\s+(?:degree|in)|M\.?Sc\.?\b|MBA\b|"
            r"MEng\b|graduate\s+degree)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "bachelors",
        re.compile(
            r"\b(?:Bachelor[''']?s\s+(?:degree|in)|BSc\b|BEng\b|"
            r"B\.?S\.?(?:c\.?)?\s+(?:in|degree)|"
            r"undergraduate\s+degree|four\s*[- ]year\s+degree|"
            r"Bachelor\s+of\s+(?:Science|Arts|Engineering))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "associates",
        re.compile(
            r"\b(?:Associate[''']?s\s+degree|two\s*[- ]year\s+degree)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "high_school",
        re.compile(
            r"\b(?:high\s+school\s+diploma|HS\s+diploma|GED)\b",
            re.IGNORECASE,
        ),
    ),
]


def _years_run(text: str) -> dict[str, Extraction]:
    if not text:
        return {}
    best: tuple[int, int | None, float] | None = None
    for m in YEARS_RE.finditer(text):
        try:
            lo = int(m.group(1))
        except (TypeError, ValueError):
            continue
        hi = None
        if m.group(2):
            try:
                hi = int(m.group(2))
            except ValueError:
                hi = None
        if not (1 <= lo <= 30):
            continue
        # confidence: closer to the start is usually a "requirements" section
        confidence = 0.85 if lo <= 25 else 0.7
        if hi is not None and hi >= lo and hi - lo <= 15:
            confidence += 0.05
        if best is None or confidence > best[2]:
            best = (lo, hi, confidence)

    if not best:
        return {}
    lo, hi, conf = best
    out = {
        "min_years_experience": Extraction(
            value=lo, confidence=conf, source="regex", rule_id="years_pattern"
        )
    }
    if hi is not None:
        out["max_years_experience"] = Extraction(
            value=hi, confidence=conf, source="regex", rule_id="years_pattern_range"
        )
    return out


def _education_run(text: str) -> dict[str, Extraction]:
    """Find the *lowest* education tier mentioned (i.e. the floor)."""
    if not text:
        return {}
    seen: list[tuple[str, int]] = []  # (tier, position)
    for tier, pattern in EDU_PATTERNS:
        for m in pattern.finditer(text):
            seen.append((tier, m.start()))
    if not seen:
        return {}
    # We emit the tier marked as "required" / "minimum" if visible, otherwise
    # the tier that appears earliest (most likely the requirement section).
    lowered = text.lower()
    for tier, _pos in seen:
        # Skip "preferred" mentions when picking the floor — those are nice-to-haves.
        # Look at the surrounding phrase.
        # This is a cheap heuristic, not a parse.
        if tier in ("phd", "masters") and "preferred" in lowered and "required" not in lowered:
            continue
        return {
            "min_education": Extraction(
                value=tier, confidence=0.8, source="regex", rule_id=f"edu_{tier}"
            )
        }
    # Fall back to first match.
    tier = seen[0][0]
    return {
        "min_education": Extraction(
            value=tier, confidence=0.7, source="regex", rule_id=f"edu_{tier}_fallback"
        )
    }


def run(text: str, title: str = "") -> dict[str, Extraction]:
    out = {}
    out.update(_years_run(text))
    out.update(_education_run(text))
    return out
