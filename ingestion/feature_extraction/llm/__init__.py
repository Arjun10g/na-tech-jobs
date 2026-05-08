"""Tier 2 LLM-based extractors. Step 1b wires NuExtract-tiny; Phase 4
reserves Qwen for hard cases."""

from ingestion.feature_extraction.llm.nuextract import NuExtract

__all__ = ["NuExtract"]
