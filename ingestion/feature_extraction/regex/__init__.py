"""Tier 1 regex extractors. Each module exports a ``run(text, title="") -> dict[str, Extraction]``."""

from ingestion.feature_extraction.regex import (
    comp_extras,
    contract_quality,
    experience_education,
    remote_schedule,
    requirements,
    salary,
    tech_stack,
)

__all__ = [
    "salary",
    "experience_education",
    "requirements",
    "remote_schedule",
    "comp_extras",
    "contract_quality",
    "tech_stack",
]
