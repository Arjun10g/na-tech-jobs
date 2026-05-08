"""Run the regex → LLM cascade and return a flat feature dict.

The orchestrator:
1. Runs all Tier 1 regex extractors over ``description_md`` + ``title``.
2. Identifies fields that came back missing or below the confidence
   threshold from :data:`confidence.TIER1_THRESHOLD`.
3. Calls Tier 2 (NuExtract) for those fields. (Stubbed in Step 1a.)
4. Merges results, with regex winning ties unless Tier 2 reports higher
   confidence on the same feature.
5. Returns a single ``dict`` of plain values plus an ``extraction_meta``
   sub-dict carrying per-field provenance.
"""

from __future__ import annotations

import logging
from typing import Any

from ingestion.feature_extraction.confidence import TIER1_THRESHOLD, Extraction
from ingestion.feature_extraction.llm.nuextract import NuExtractStub
from ingestion.feature_extraction.regex import (
    comp_extras,
    contract_quality,
    experience_education,
    remote_schedule,
    requirements,
    salary,
    tech_stack,
)

logger = logging.getLogger("feature_extraction.cascade")

# All fields the cascade can in principle return. Used to detect "Tier 2 should
# try this" vs "the feature genuinely doesn't apply."
ALL_FEATURE_FIELDS: tuple[str, ...] = (
    # salary
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
    "salary_disclosed",
    # experience / education
    "min_years_experience",
    "max_years_experience",
    "min_education",
    # requirements
    "requires_security_clearance",
    "clearance_level",
    "requires_citizenship",
    "offers_visa_sponsorship",
    # remote / schedule
    "remote_policy_extracted",
    "on_call_required",
    "max_travel_percent",
    # comp extras
    "offers_equity",
    "equity_form",
    "bonus_mentioned",
    "bonus_type",
    "offers_relocation",
    # contract / quality / language / manager
    "contract_type",
    "posting_quality",
    "language_requirements",
    "manager_role",
    "direct_reports_count",
    # tech stack
    "tech_stack",
)

REGEX_MODULES = (
    salary,
    experience_education,
    requirements,
    remote_schedule,
    comp_extras,
    contract_quality,
    tech_stack,
)


# Fields that Tier 2 should attempt when regex didn't fill them. Some fields
# (salary_disclosed, posting_quality default) shouldn't escalate to LLM.
LLM_ELIGIBLE_FIELDS: frozenset[str] = frozenset(
    {
        "min_years_experience",
        "min_education",
        "requires_security_clearance",
        "clearance_level",
        "requires_citizenship",
        "offers_visa_sponsorship",
        "remote_policy_extracted",
        "on_call_required",
        "offers_equity",
        "bonus_mentioned",
        "tech_stack",
        "industry_experience",
        "team_or_department",
    }
)


_llm_singleton: NuExtractStub | None = None


def _get_llm() -> NuExtractStub:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = NuExtractStub()
    return _llm_singleton


def _merge(accumulator: dict[str, Extraction], updates: dict[str, Extraction]) -> None:
    """Merge ``updates`` into ``accumulator``. Higher confidence wins ties."""
    for name, extraction in updates.items():
        existing = accumulator.get(name)
        if existing is None or extraction.confidence > existing.confidence:
            accumulator[name] = extraction


def extract_features(
    description_md: str | None,
    title: str = "",
    *,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Run the full cascade. Returns plain values + ``extraction_meta``.

    Plain values are flat at the top level (so callers can ``dict.update``
    them onto a CanonicalJob). Per-field provenance lives in
    ``result["extraction_meta"]``.
    """
    text = description_md or ""
    title = title or ""

    accumulator: dict[str, Extraction] = {}

    # Tier 1: regex modules.
    for module in REGEX_MODULES:
        try:
            updates = module.run(text, title)
        except Exception as exc:  # noqa: BLE001
            logger.warning("regex module %s raised: %s", module.__name__, exc)
            continue
        _merge(accumulator, updates)

    # Identify Tier 2 candidates.
    if use_llm:
        missing = [
            f
            for f in LLM_ELIGIBLE_FIELDS
            if (f not in accumulator) or accumulator[f].confidence < TIER1_THRESHOLD
        ]
        if missing:
            try:
                llm_updates = _get_llm().run(text, title, missing)
                _merge(accumulator, llm_updates)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM tier raised: %s", exc)

    # Materialise output.
    out: dict[str, Any] = {}
    meta: dict[str, dict[str, Any]] = {}
    for name, extraction in accumulator.items():
        out[name] = extraction.value
        meta[name] = extraction.as_meta()

    out["extraction_meta"] = meta
    out["extraction_version"] = "v1"
    return out
