"""Tier 2/3 LLM-based extractors. Phase 1a ships only the stub interfaces;
Phase 1b wires NuExtract; Phase 4 reserves Qwen for hard cases."""

from ingestion.feature_extraction.llm.nuextract import NuExtractStub

__all__ = ["NuExtractStub"]
