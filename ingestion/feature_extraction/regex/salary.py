"""Salary range mining from job descriptions.

Matches "$135,000 - $180,000 USD", "$135K - $180K", "USD 135,000 to 180,000",
and a handful of CAD variants. Confidence is high only when both ends of the
range are present, the values are in a plausible NA tech-salary band
(USD 30k-1M / CAD 40k-1.3M / equivalent hourly), and the match is anchored
by a salary keyword within ±100 chars.
"""

from __future__ import annotations

import re

from ingestion.feature_extraction.confidence import Extraction

# ── primitive number tokens ────────────────────────────────────────────────
# Captures things like "180,000", "180000", "180k", "180K", "180.0k", "22.50".
_NUM = r"\d{1,3}(?:[,.]?\d{3})*(?:\.\d{1,2})?(?:\s?[KkMm]\b)?"

# ── currency tokens ────────────────────────────────────────────────────────
_CCY_BEFORE = r"(?:\$\s?(?:USD|CAD|US|CDN)?\s?|USD\s?\$?\s?|CAD\s?\$?\s?|US\$\s?|CA\$\s?|C\$\s?)"
_CCY_AFTER = r"(?:\s?(?:USD|CAD|US|CDN|US\s?dollars?|Canadian\s?dollars?))?"
_RANGE_SEP = r"\s*(?:[-–—]|to|–|—)\s*"
# The second value may omit a currency prefix ("$135K - $180K USD" vs
# "USD 130,000 to 175,000"). Make the second prefix optional.
_CCY_BEFORE_OPT = f"(?:{_CCY_BEFORE})?"

# Anchor keywords that should sit within ~100 chars of the matched range.
SALARY_ANCHOR_RE = re.compile(
    r"\b(?:salary|compensation|pay\s+range|base\s+pay|base\s+salary|annual\s+(?:base|salary|pay)"
    r"|target\s+(?:earnings|pay)|expected\s+(?:salary|compensation)|wage|hourly\s+rate|range\s+is)\b",
    re.IGNORECASE,
)

# Period anchors near the value.
PERIOD_NEAR_RE = re.compile(
    r"\b(?:per\s+(?:year|hour|month|annum)|annually|/\s?yr|/\s?year|/\s?hr|/\s?hour|annual)\b",
    re.IGNORECASE,
)

# Composite range pattern: matches ($A-$B) variants. Stays *case-insensitive*.
RANGE_RE = re.compile(
    rf"{_CCY_BEFORE}({_NUM}){_CCY_AFTER}{_RANGE_SEP}{_CCY_BEFORE_OPT}({_NUM}){_CCY_AFTER}",
    re.IGNORECASE,
)


def _to_float(token: str) -> float | None:
    s = token.strip().replace(",", "").replace(" ", "")
    multiplier = 1.0
    if s.lower().endswith("k"):
        multiplier = 1_000.0
        s = s[:-1]
    elif s.lower().endswith("m"):
        multiplier = 1_000_000.0
        s = s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return None


def _detect_currency(window: str) -> str:
    upper = window.upper()
    if "CAD" in upper or "CDN" in upper or "C$" in upper or "CA$" in upper or "CANADIAN" in upper:
        return "CAD"
    return "USD"


def _detect_period(window: str, value: float) -> str:
    """Resolve the salary period from anchor words first; fall back on magnitude."""
    lower = window.lower()
    is_hourly_signal = "hour" in lower or "/hr" in lower or ("/yr" not in lower and "/h" in lower)
    if is_hourly_signal and "annual" not in lower and "per year" not in lower:
        return "hour"
    if "month" in lower:
        return "month"
    if "day" in lower:
        return "day"
    # Magnitude guard: if the value is way too small for an annual NA salary,
    # call it hourly.
    if value < 5_000:
        return "hour"
    return "year"


def _plausible(period: str, currency: str, value: float) -> bool:
    if period == "year":
        return 30_000 <= value <= 1_500_000
    if period == "month":
        return 2_000 <= value <= 100_000
    if period == "day":
        return 200 <= value <= 4_000
    if period == "hour":
        return 10 <= value <= 500
    return False


def run(text: str, title: str = "") -> dict[str, Extraction]:
    """Mine a salary range from ``text`` (markdown). Returns extractions for
    salary_min, salary_max, salary_currency, salary_period, salary_disclosed.
    Empty dict if nothing usable found."""
    if not text:
        return {}

    best: tuple[float, dict] | None = None
    for m in RANGE_RE.finditer(text):
        a, b = _to_float(m.group(1)), _to_float(m.group(2))
        if a is None or b is None:
            continue
        if a > b:
            a, b = b, a
        # context window for anchor / currency / period detection
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 100)
        window = text[start:end]

        # Hard requirement: anchor word in window, otherwise skip.
        anchor = SALARY_ANCHOR_RE.search(window)
        if not anchor:
            continue

        currency = _detect_currency(window)
        period = _detect_period(window, b)
        if not _plausible(period, currency, b):
            continue

        # Score: prefer matches whose anchor word is closer + within reasonable
        # range of magnitudes. Year > hour for typical NA listings.
        anchor_distance = min(abs(anchor.start() - (m.start() - start)), 100)
        score = (1.0 if period == "year" else 0.7) * (1 - anchor_distance / 200)
        candidate = {
            "salary_min": a,
            "salary_max": b,
            "salary_currency": currency,
            "salary_period": period,
            "rule_id": "salary_range_anchored",
        }
        if best is None or score > best[0]:
            best = (score, candidate)

    if not best:
        return {}

    score, c = best
    confidence = min(0.95, 0.7 + score * 0.25)
    return {
        "salary_min": Extraction(
            value=c["salary_min"], confidence=confidence, source="regex", rule_id=c["rule_id"]
        ),
        "salary_max": Extraction(
            value=c["salary_max"], confidence=confidence, source="regex", rule_id=c["rule_id"]
        ),
        "salary_currency": Extraction(
            value=c["salary_currency"],
            confidence=confidence,
            source="regex",
            rule_id=c["rule_id"],
        ),
        "salary_period": Extraction(
            value=c["salary_period"], confidence=confidence, source="regex", rule_id=c["rule_id"]
        ),
        "salary_disclosed": Extraction(
            value=True, confidence=confidence, source="regex", rule_id=c["rule_id"]
        ),
    }
