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
from ingestion.feature_extraction.llm.nuextract import NuExtract
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


# LLM tier is wired but DORMANT by default (frozenset()). Rationale: the
# benchmarks at Step 1b showed a 12+ hour backfill on Apple-MPS for
# coverage gains that don't feed the salary regressor (Step 3) or any other
# downstream consumer yet. The bge-m3 description embedding in Phase 5 also
# carries most of the same semantic signal, making structured `tech_stack`
# / `industry_experience` columns redundant at the v1 demo.
#
# Re-enable by populating this set when a downstream consumer needs structured
# values. Suggested fields, with the regex coverage they'd backfill:
#   "min_education"            # regex 25%; LLM picks up prose phrasings
#   "requires_citizenship"     # regex 14%; ITAR / federal-contractor varied
#   "offers_visa_sponsorship"  # regex 0.3%; usually contextual
#   "tech_stack"               # regex 64%; LLM extends with prose-only skills
#   "industry_experience"      # regex 0%; LLM-only
#   "team_or_department"       # regex 0%; LLM-only
LLM_ELIGIBLE_FIELDS: frozenset[str] = frozenset()


_llm_singleton: NuExtract | None = None


def _get_llm() -> NuExtract:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = NuExtract()
    return _llm_singleton


def _merge(accumulator: dict[str, Extraction], updates: dict[str, Extraction]) -> None:
    """Merge ``updates`` into ``accumulator``. Higher confidence wins ties."""
    for name, extraction in updates.items():
        existing = accumulator.get(name)
        if existing is None or extraction.confidence > existing.confidence:
            accumulator[name] = extraction


def _run_regex(text: str, title: str) -> dict[str, Extraction]:
    """Tier 1 only — runs every regex module and merges the results."""
    accumulator: dict[str, Extraction] = {}
    for module in REGEX_MODULES:
        try:
            updates = module.run(text, title)
        except Exception as exc:  # noqa: BLE001
            logger.warning("regex module %s raised: %s", module.__name__, exc)
            continue
        _merge(accumulator, updates)
    return accumulator


def _missing_for_llm(accumulator: dict[str, Extraction]) -> list[str]:
    return [
        f
        for f in LLM_ELIGIBLE_FIELDS
        if (f not in accumulator) or accumulator[f].confidence < TIER1_THRESHOLD
    ]


def _materialize(accumulator: dict[str, Extraction]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    meta: dict[str, dict[str, Any]] = {}
    for name, extraction in accumulator.items():
        out[name] = extraction.value
        meta[name] = extraction.as_meta()
    out["extraction_meta"] = meta
    out["extraction_version"] = "v1"
    return out


def extract_features(
    description_md: str | None,
    title: str = "",
    *,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Run the full cascade on a single description. Returns plain values +
    ``extraction_meta``."""
    text = description_md or ""
    title = title or ""

    accumulator = _run_regex(text, title)

    if use_llm:
        missing = _missing_for_llm(accumulator)
        if missing:
            try:
                llm_updates = _get_llm().run(text, title, missing)
                _merge(accumulator, llm_updates)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM tier raised: %s", exc)

    return _materialize(accumulator)


def extract_features_batch(
    rows: list[tuple[str | None, str]],
    *,
    use_llm: bool = True,
    llm_batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Vectorised cascade for a list of ``(description_md, title)`` pairs.

    Big throughput win when ``use_llm=True``: regex runs per-row (cheap),
    then all rows that still have missing LLM-eligible fields are pushed
    through ``NuExtract.run_batch`` ``llm_batch_size`` at a time. On Apple
    MPS this gives a ~3-5x speedup over the per-row path. The cost: peak
    memory rises with batch size since left-padding pads to the longest
    prompt in the batch.
    """
    accumulators: list[dict[str, Extraction]] = []
    for description_md, title in rows:
        text = description_md or ""
        accumulators.append(_run_regex(text, title or ""))

    if use_llm:
        # Build the LLM work list: only rows with at least one missing
        # eligible field that also have non-empty text.
        llm_jobs: list[tuple[int, str, str, list[str]]] = []
        for i, (description_md, title) in enumerate(rows):
            text = description_md or ""
            if not text:
                continue
            missing = _missing_for_llm(accumulators[i])
            if missing:
                llm_jobs.append((i, text, title or "", missing))

        if llm_jobs:
            llm = _get_llm()
            for chunk_start in range(0, len(llm_jobs), llm_batch_size):
                chunk = llm_jobs[chunk_start : chunk_start + llm_batch_size]
                items = [(text, title, missing) for _, text, title, missing in chunk]
                try:
                    chunk_results = llm.run_batch(items)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("LLM batch raised: %s", exc)
                    continue
                for (orig_idx, _, _, _), updates in zip(chunk, chunk_results, strict=True):
                    if updates:
                        _merge(accumulators[orig_idx], updates)

    return [_materialize(acc) for acc in accumulators]
