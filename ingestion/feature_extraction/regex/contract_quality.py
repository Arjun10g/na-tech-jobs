"""Contract type, posting quality, language requirements, manager-role,
direct-reports-count extractors. Bundled because most are short patterns."""

from __future__ import annotations

import re

from ingestion.feature_extraction.confidence import Extraction

# ── Contract type ──────────────────────────────────────────────────────────
CONTRACT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "internship",
        re.compile(
            r"\b(?:internship|intern\b|co-?op\s+student|co-?op\s+position|summer\s+intern)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "contract",
        re.compile(
            r"\b(?:contractor|contract\s+(?:role|position|to\s+hire)|fixed[- ]term\s+contract|FTC\b|"
            r"consultancy\s+role|temporary\s+contract|contract\s+basis|contract\s+work)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "temporary",
        re.compile(r"\btemporary\s+(?:role|position)\b|\btemp\s+role\b", re.IGNORECASE),
    ),
    (
        "part_time",
        re.compile(r"\bpart[- ]time\b", re.IGNORECASE),
    ),
    (
        "full_time",
        re.compile(r"\bfull[- ]time\b|\bpermanent\s+(?:role|position|hire)\b", re.IGNORECASE),
    ),
]

# ── Posting quality (real / evergreen / talent_community) ──────────────────
EVERGREEN_TITLE_RE = re.compile(
    r"^(?:future\s+opportunit|general\s+application|join\s+our\s+talent|"
    r"talent\s+(?:pool|community|network)|expression\s+of\s+interest|"
    r"calling\s+all|always\s+hiring|future\s+hires?|stay\s+in\s+touch)",
    re.IGNORECASE,
)
TALENT_COMMUNITY_TITLE_RE = re.compile(
    r"\btalent\s+(?:network|community|pool)\b|\bnewsletter\s+sign[- ]up\b",
    re.IGNORECASE,
)

# ── Language requirements ──────────────────────────────────────────────────
LANG_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "en",
        re.compile(r"\b(?:fluent|proficient|native)\s+(?:English|in\s+English)\b", re.IGNORECASE),
    ),
    (
        "fr",
        re.compile(
            r"\b(?:fluent|proficient|native|bilingual)\s+(?:French|in\s+French)\b|\bbilingual\s+\(English/French\)|\bEnglish/French\s+bilingual\b",
            re.IGNORECASE,
        ),
    ),
    (
        "es",
        re.compile(r"\b(?:fluent|proficient|native)\s+(?:Spanish|in\s+Spanish)\b", re.IGNORECASE),
    ),
    (
        "ja",
        re.compile(
            r"\b(?:fluent|native)\s+(?:Japanese|in\s+Japanese)\b|日本語|流暢", re.IGNORECASE
        ),
    ),
    (
        "zh",
        re.compile(
            r"\b(?:fluent|proficient|native)\s+(?:Mandarin|Chinese|Cantonese)\b", re.IGNORECASE
        ),
    ),
]

# ── Manager role / direct reports ──────────────────────────────────────────
MANAGER_TITLE_RE = re.compile(
    r"\b(?:Manager|Director|VP|Vice\s+President|Head\s+of)\b", re.IGNORECASE
)
SENIOR_MGR_TITLE_RE = re.compile(r"\b(?:Senior\s+Manager|Sr\.?\s+Manager)\b", re.IGNORECASE)
DIRECTOR_TITLE_RE = re.compile(r"\b(?:Director|Head\s+of)\b", re.IGNORECASE)
EXEC_TITLE_RE = re.compile(r"\b(?:VP|Vice\s+President|Chief\s+\w+|C[EFOTPI]O)\b", re.IGNORECASE)
LEAD_TITLE_RE = re.compile(
    r"\b(?:Tech(?:nical)?\s+Lead|Lead\s+Engineer|Staff\s+Engineer|Lead\s+Scientist)\b",
    re.IGNORECASE,
)

DIRECT_REPORTS_RE = re.compile(
    r"\b(?:manage|leading|oversee|supervis(?:ing|e)|responsible\s+for)\s+(?:a\s+)?(?:team\s+of\s+)?(\d{1,3})\s+(?:direct\s+reports?|engineers|scientists|people|staff|associates)",
    re.IGNORECASE,
)


def run(text: str, title: str = "") -> dict[str, Extraction]:
    out: dict[str, Extraction] = {}

    # ── Contract type ──────────────────────────────────────────────────────
    text_or_title = (title or "") + "\n" + (text or "")
    for ctype, pattern in CONTRACT_PATTERNS:
        if pattern.search(text_or_title):
            out["contract_type"] = Extraction(
                value=ctype,
                confidence=0.8 if ctype != "full_time" else 0.7,
                source="regex",
                rule_id=f"contract_{ctype}",
            )
            break

    # ── Posting quality ────────────────────────────────────────────────────
    if title and EVERGREEN_TITLE_RE.match(title.strip()):
        out["posting_quality"] = Extraction(
            value="evergreen_pool", confidence=0.85, source="regex", rule_id="evergreen_title"
        )
    elif title and TALENT_COMMUNITY_TITLE_RE.search(title):
        out["posting_quality"] = Extraction(
            value="talent_community",
            confidence=0.85,
            source="regex",
            rule_id="talent_community_title",
        )
    else:
        out["posting_quality"] = Extraction(
            value="real", confidence=0.6, source="regex", rule_id="default_real"
        )

    # ── Language requirements ──────────────────────────────────────────────
    langs: list[str] = []
    for code, pattern in LANG_PATTERNS:
        if pattern.search(text):
            langs.append(code)
    if langs:
        out["language_requirements"] = Extraction(
            value=langs, confidence=0.8, source="regex", rule_id="lang_pattern"
        )

    # ── Manager role ───────────────────────────────────────────────────────
    # Only emit when there's an actual title-level signal. Default to None
    # (i.e. unknown) so consumers can distinguish "no signal" from "IC".
    if title:
        title_clean = title.strip()
        if EXEC_TITLE_RE.search(title_clean):
            out["manager_role"] = Extraction(
                value="exec", confidence=0.9, source="regex", rule_id="exec_title"
            )
        elif DIRECTOR_TITLE_RE.search(title_clean):
            out["manager_role"] = Extraction(
                value="director", confidence=0.9, source="regex", rule_id="director_title"
            )
        elif SENIOR_MGR_TITLE_RE.search(title_clean):
            out["manager_role"] = Extraction(
                value="senior_manager", confidence=0.9, source="regex", rule_id="sr_mgr_title"
            )
        elif MANAGER_TITLE_RE.search(title_clean):
            out["manager_role"] = Extraction(
                value="manager", confidence=0.85, source="regex", rule_id="mgr_title"
            )
        elif LEAD_TITLE_RE.search(title_clean):
            out["manager_role"] = Extraction(
                value="tech_lead", confidence=0.85, source="regex", rule_id="lead_title"
            )

    # ── Direct reports count ───────────────────────────────────────────────
    m = DIRECT_REPORTS_RE.search(text)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 200:
                out["direct_reports_count"] = Extraction(
                    value=n, confidence=0.7, source="regex", rule_id="direct_reports_pattern"
                )
        except ValueError:
            pass

    return out
