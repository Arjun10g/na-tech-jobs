"""Regex-first cascade with NuExtract LLM fallback for job-description features.

Public surface: ``extract_features(description_md, title="") -> dict``. Returns
a dict of feature values plus an ``extraction_meta`` dict carrying per-field
provenance (source: regex/llm/structured, confidence, rule_id).

See CLAUDE.md and MAINTENANCE.md for the design rationale.
"""

from ingestion.feature_extraction.cascade import extract_features

__all__ = ["extract_features"]
