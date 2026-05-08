"""Hard-requirement extractors: security clearance, citizenship, sponsorship."""

from __future__ import annotations

import re

from ingestion.feature_extraction.confidence import Extraction

# ── Security clearance ─────────────────────────────────────────────────────
CLEARANCE_RE = re.compile(
    r"\b(?:active\s+)?(?P<level>TS\s*/\s*SCI|TS/SCI|Top\s*Secret(?:\s*/\s*SCI)?"
    r"|Secret(?:\s+clearance)?|Confidential\s+clearance|Public\s+Trust)\b",
    re.IGNORECASE,
)
CLEARANCE_NEG_RE = re.compile(
    r"\b(?:not\s+required|no\s+clearance\s+required|able\s+to\s+obtain)\b",
    re.IGNORECASE,
)

CLEARANCE_LEVEL_MAP: dict[str, str] = {
    "ts/sci": "ts_sci",
    "ts / sci": "ts_sci",
    "top secret/sci": "ts_sci",
    "top secret / sci": "ts_sci",
    "top secret": "top_secret",
    "secret clearance": "secret",
    "secret": "secret",
    "confidential clearance": "confidential",
    "public trust": "public_trust",
}

# ── Citizenship ────────────────────────────────────────────────────────────
US_CITIZENSHIP_RE = re.compile(
    r"\b(?:U\.?S\.?\s+citizens?(?:hip\s+is\s+required)?|"
    r"American\s+citizen|"
    r"must\s+be\s+(?:a\s+)?U\.?S\.?\s+citizen|"
    r"requires?\s+(?:a\s+)?U\.?S\.?\s+(?:person|citizen)|"
    r"due\s+to\s+(?:ITAR|EAR|export\s+control|federal\s+contracting))\b",
    re.IGNORECASE,
)
CA_CITIZENSHIP_RE = re.compile(
    r"\b(?:Canadian\s+citizen(?:ship)?(?:\s+is\s+required)?|"
    r"must\s+be\s+(?:a\s+)?Canadian\s+citizen|"
    r"Canadian\s+permanent\s+resident)\b",
    re.IGNORECASE,
)

# ── Visa sponsorship ───────────────────────────────────────────────────────
SPONSORSHIP_YES_RE = re.compile(
    r"\b(?:we\s+(?:offer|provide)\s+visa\s+sponsorship|"
    r"visa\s+sponsorship\s+(?:is\s+)?(?:available|offered|provided)|"
    r"sponsorship\s+for\s+work\s+authorization|"
    r"H-?1B\s+(?:transfers?\s+welcome|sponsorship\s+available))\b",
    re.IGNORECASE,
)
SPONSORSHIP_NO_RE = re.compile(
    r"\b(?:not?\s+(?:able\s+to\s+|in\s+a\s+position\s+to\s+)?(?:offer|provide|sponsor)\s+(?:visa|H-?1B)|"
    r"visa\s+sponsorship\s+(?:is\s+)?not\s+(?:available|offered|provided)|"
    r"unable\s+to\s+sponsor|"
    r"do\s+not\s+sponsor|"
    r"this\s+position\s+is\s+not\s+eligible\s+for\s+visa\s+sponsorship|"
    r"(?:must\s+be|requires?)\s+(?:legally\s+)?authorized\s+to\s+work\s+(?:in\s+(?:the\s+)?(?:US|U\.?S\.?A?))?\s*without\s+sponsorship)",
    re.IGNORECASE,
)


def run(text: str, title: str = "") -> dict[str, Extraction]:
    out: dict[str, Extraction] = {}
    if not text:
        return out

    # ── Security clearance ─────────────────────────────────────────────────
    clearance_match = CLEARANCE_RE.search(text)
    if clearance_match and not CLEARANCE_NEG_RE.search(
        text[max(0, clearance_match.start() - 80) : clearance_match.end() + 30]
    ):
        raw = clearance_match.group("level").lower().strip()
        # normalize whitespace
        normalized = re.sub(r"\s+", " ", raw).replace(" / ", "/")
        level = CLEARANCE_LEVEL_MAP.get(normalized)
        if level is None:
            for key, v in CLEARANCE_LEVEL_MAP.items():
                if key in normalized:
                    level = v
                    break
        out["requires_security_clearance"] = Extraction(
            value=True, confidence=0.92, source="regex", rule_id="clearance_pattern"
        )
        if level:
            out["clearance_level"] = Extraction(
                value=level, confidence=0.92, source="regex", rule_id="clearance_level"
            )

    # ── Citizenship ────────────────────────────────────────────────────────
    countries: list[str] = []
    if US_CITIZENSHIP_RE.search(text):
        countries.append("US")
    if CA_CITIZENSHIP_RE.search(text):
        countries.append("CA")
    if countries:
        out["requires_citizenship"] = Extraction(
            value=countries, confidence=0.88, source="regex", rule_id="citizenship_pattern"
        )

    # ── Sponsorship ────────────────────────────────────────────────────────
    if SPONSORSHIP_NO_RE.search(text):
        out["offers_visa_sponsorship"] = Extraction(
            value="no", confidence=0.9, source="regex", rule_id="sponsorship_no"
        )
    elif SPONSORSHIP_YES_RE.search(text):
        out["offers_visa_sponsorship"] = Extraction(
            value="yes", confidence=0.9, source="regex", rule_id="sponsorship_yes"
        )

    return out
