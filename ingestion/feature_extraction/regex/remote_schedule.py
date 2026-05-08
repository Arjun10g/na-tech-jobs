"""Remote/hybrid/onsite policy + on-call + travel-percent extractors.

The base location parser already populates ``remote_policy`` from
``location_raw`` (CLAUDE.md §6). This module *augments* that signal by reading
the description, since most ATSes don't put remote/hybrid info in the
location string. Fill rate jumped from ~19% to typically 60-80% in spot checks.
"""

from __future__ import annotations

import re

from ingestion.feature_extraction.confidence import Extraction

REMOTE_RE = re.compile(
    r"\b(?:fully\s+remote|100%\s+remote|remote[- ]first|remote\s+(?:role|position|opportunity)|"
    r"work\s+from\s+(?:home|anywhere)|distributed\s+team)\b",
    re.IGNORECASE,
)
HYBRID_RE = re.compile(
    r"\b(?:hybrid(?:\s+(?:role|position|work|schedule))?|"
    r"\d\s+days?\s+(?:per\s+week\s+)?(?:in\s+(?:the\s+)?office|onsite|in[- ]person))\b",
    re.IGNORECASE,
)
ONSITE_RE = re.compile(
    r"\b(?:on[- ]?site\s+(?:only|role|position)|in[- ]office\s+(?:role|position|only)|"
    r"this\s+is\s+(?:a\s+)?on[- ]?site|fully\s+in[- ]office|five\s+days\s+(?:per\s+week|/\s?week)\s+in\s+office)\b",
    re.IGNORECASE,
)

ON_CALL_RE = re.compile(
    r"\b(?:on[- ]call(?:\s+rotation|\s+responsibilities)?|"
    r"24/7\s+(?:on[- ]call|support|coverage)|"
    r"pager\s+rotation|incident\s+response\s+rotation)\b",
    re.IGNORECASE,
)
ON_CALL_NEG_RE = re.compile(
    r"\b(?:no\s+on[- ]call|not\s+on[- ]call)\b",
    re.IGNORECASE,
)

# Require "travel" within ~30 chars of the percentage to avoid matching
# "up to 100% match on 401(k)" and similar benefits language.
TRAVEL_PCT_RE = re.compile(
    r"(?:travel\s+(?:up\s+to\s+)?(?P<pct1>\d{1,3})\s*%"
    r"|(?:up\s+to\s+)?(?P<pct2>\d{1,3})\s*%\s+travel"
    r"|(?P<pct3>\d{1,3})\s*%\s+of\s+the\s+time\s+(?:travel|on\s+the\s+road))",
    re.IGNORECASE,
)
TRAVEL_NEG_RE = re.compile(r"\bno\s+travel\b|\b0%\s*travel\b", re.IGNORECASE)


def run(text: str, title: str = "") -> dict[str, Extraction]:
    out: dict[str, Extraction] = {}
    if not text:
        return out

    # Remote / hybrid / onsite — first match wins; check most-specific first.
    if HYBRID_RE.search(text):
        out["remote_policy_extracted"] = Extraction(
            value="hybrid", confidence=0.9, source="regex", rule_id="hybrid_keyword"
        )
    elif REMOTE_RE.search(text):
        out["remote_policy_extracted"] = Extraction(
            value="remote", confidence=0.85, source="regex", rule_id="remote_keyword"
        )
    elif ONSITE_RE.search(text):
        out["remote_policy_extracted"] = Extraction(
            value="onsite", confidence=0.85, source="regex", rule_id="onsite_keyword"
        )

    # On-call.
    if ON_CALL_NEG_RE.search(text):
        out["on_call_required"] = Extraction(
            value=False, confidence=0.85, source="regex", rule_id="on_call_negated"
        )
    elif ON_CALL_RE.search(text):
        out["on_call_required"] = Extraction(
            value=True, confidence=0.85, source="regex", rule_id="on_call_pattern"
        )

    # Travel percent.
    if TRAVEL_NEG_RE.search(text):
        out["max_travel_percent"] = Extraction(
            value=0, confidence=0.9, source="regex", rule_id="travel_zero"
        )
    else:
        match = TRAVEL_PCT_RE.search(text)
        if match:
            pct_str = match.group("pct1") or match.group("pct2") or match.group("pct3")
            try:
                pct = int(pct_str)
                if 0 <= pct <= 100:
                    out["max_travel_percent"] = Extraction(
                        value=pct, confidence=0.85, source="regex", rule_id="travel_percent"
                    )
            except (TypeError, ValueError):
                pass

    return out
