"""Equity, bonus, relocation extractors."""

from __future__ import annotations

import re

from ingestion.feature_extraction.confidence import Extraction

# Equity
EQUITY_RE = re.compile(
    r"\b(?:equity(?:\s+(?:package|grant|compensation))?|"
    r"stock\s+options?|RSUs?\b|restricted\s+stock\s+units?|"
    r"profit\s*sharing|ownership\s+stake|equity\s+component)\b",
    re.IGNORECASE,
)
EQUITY_RSU_RE = re.compile(r"\bRSUs?\b|\brestricted\s+stock\s+units?\b", re.IGNORECASE)
EQUITY_OPTIONS_RE = re.compile(r"\bstock\s+options?\b|\bequity\s+options?\b", re.IGNORECASE)
EQUITY_PROFIT_RE = re.compile(r"\bprofit[\s-]?sharing\b", re.IGNORECASE)

# Bonus
BONUS_RE = re.compile(
    r"\b(?:bonus(?:es)?|sign(?:ing|-on)\s+bonus|annual\s+bonus|"
    r"performance\s+bonus|retention\s+bonus|incentive\s+(?:pay|bonus)|"
    r"target\s+bonus)\b",
    re.IGNORECASE,
)
BONUS_SIGNING_RE = re.compile(r"\bsign(?:ing|-on)\s+bonus\b|\bsigning[- ]bonus\b", re.IGNORECASE)
BONUS_PERF_RE = re.compile(
    r"\bperformance\s+bonus\b|\bperformance[- ]based\s+bonus\b", re.IGNORECASE
)
BONUS_RETENTION_RE = re.compile(r"\bretention\s+bonus\b", re.IGNORECASE)
BONUS_ANNUAL_RE = re.compile(r"\bannual\s+(?:bonus|incentive)\b|\btarget\s+bonus\b", re.IGNORECASE)

# Relocation
RELOC_RE = re.compile(
    r"\b(?:relocation\s+(?:assistance|support|package|allowance|benefits)|"
    r"will\s+(?:help|assist)\s+with\s+relocation|"
    r"we\s+(?:offer|provide)\s+relocation)\b",
    re.IGNORECASE,
)
RELOC_NEG_RE = re.compile(
    r"\b(?:no\s+relocation|not\s+offer\s+relocation|relocation\s+not\s+(?:available|offered))\b",
    re.IGNORECASE,
)


def run(text: str, title: str = "") -> dict[str, Extraction]:
    out: dict[str, Extraction] = {}
    if not text:
        return out

    # Equity.
    if EQUITY_RE.search(text):
        out["offers_equity"] = Extraction(
            value=True, confidence=0.85, source="regex", rule_id="equity_keyword"
        )
        # form (most specific wins)
        if EQUITY_RSU_RE.search(text):
            form = "rsu"
        elif EQUITY_OPTIONS_RE.search(text):
            form = "options"
        elif EQUITY_PROFIT_RE.search(text):
            form = "profit_sharing"
        else:
            form = "other"
        out["equity_form"] = Extraction(
            value=form, confidence=0.85, source="regex", rule_id=f"equity_form_{form}"
        )

    # Bonus.
    if BONUS_RE.search(text):
        out["bonus_mentioned"] = Extraction(
            value=True, confidence=0.85, source="regex", rule_id="bonus_keyword"
        )
        if BONUS_SIGNING_RE.search(text):
            btype = "signing"
        elif BONUS_RETENTION_RE.search(text):
            btype = "retention"
        elif BONUS_PERF_RE.search(text):
            btype = "performance"
        elif BONUS_ANNUAL_RE.search(text):
            btype = "annual"
        else:
            btype = None
        if btype:
            out["bonus_type"] = Extraction(
                value=btype, confidence=0.85, source="regex", rule_id=f"bonus_{btype}"
            )

    # Relocation.
    if RELOC_NEG_RE.search(text):
        out["offers_relocation"] = Extraction(
            value=False, confidence=0.85, source="regex", rule_id="relocation_negated"
        )
    elif RELOC_RE.search(text):
        out["offers_relocation"] = Extraction(
            value=True, confidence=0.85, source="regex", rule_id="relocation_keyword"
        )

    return out
