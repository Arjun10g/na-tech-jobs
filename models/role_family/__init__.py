"""Role-family classifier (DeBERTa-v3-base + LoRA, weakly supervised).

Replaces the title-regex heuristic in ``ingestion/normalize.py`` for
downstream consumers (Phase 4 ``curated/enrich.py``, Phase 5 RAG payload
filters).
"""
