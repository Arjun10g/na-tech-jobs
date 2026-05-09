"""Lazy singleton retriever for the Gradio app.

The first call loads:
- the Qdrant local-mode client (~10 ms),
- the embedder (MiniLM by default; swap to bge-m3 by setting
  ``RAG_EMBEDDER`` env var),
- optionally the cross-encoder reranker (``RAG_RERANKER`` env var to
  enable; defaults to OFF for cold-start latency).
- a parent-chunk lookup built from the enriched curated parquet.

After construction we cache and reuse for every query.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("app.retriever_loader")

_retriever_singleton: Any | None = None
_qdrant_path: Path = Path("data/qdrant")


def _resolve_qdrant_path() -> Path:
    env = os.environ.get("RAG_QDRANT_PATH")
    return Path(env) if env else _qdrant_path


def _resolve_embedder_kind() -> tuple[bool, str | None]:
    """Returns (lite, model_name_override)."""
    name = os.environ.get("RAG_EMBEDDER")
    if not name or name == "lite":
        return True, None
    if "MiniLM" in name or "minilm" in name:
        return True, name
    return False, name


def _resolve_reranker_kind() -> tuple[bool, str | None] | None:
    """Returns None to skip reranker, or (lite, model_name_override)."""
    flag = os.environ.get("RAG_RERANKER", "off").lower()
    if flag in ("off", "false", "0", ""):
        return None
    if flag in ("on", "true", "1", "lite"):
        return True, None
    if "MiniLM" in flag:
        return True, flag
    return False, flag


def get_retriever():
    """Return a cached :class:`rag.pipeline.HybridRetriever`."""
    global _retriever_singleton
    if _retriever_singleton is not None:
        return _retriever_singleton

    from rag.embedder import load_embedder
    from rag.pipeline import HybridRetriever, build_parent_lookup
    from rag.qdrant_client import COLLECTION_DENSE, get_client

    qdrant_path = _resolve_qdrant_path()
    if not qdrant_path.exists():
        raise FileNotFoundError(
            f"Qdrant index missing at {qdrant_path}. "
            "Run `uv run python -m scripts.index_jobs --lite` first."
        )

    logger.info("opening qdrant local client at %s", qdrant_path)
    client = get_client(qdrant_path)
    if not client.collection_exists(COLLECTION_DENSE):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_DENSE}' does not exist at {qdrant_path}. "
            "Re-run `scripts.index_jobs`."
        )

    lite, override = _resolve_embedder_kind()
    embedder = load_embedder(override, lite=lite)
    logger.info("embedder ready :: %s (dim=%d)", embedder.__class__.__name__, embedder.dense_dim)

    reranker = None
    rerank_cfg = _resolve_reranker_kind()
    if rerank_cfg is not None:
        from rag.reranker import load_reranker

        rlite, roverride = rerank_cfg
        reranker = load_reranker(roverride, lite=rlite)
        logger.info("reranker ready :: %s", reranker.__class__.__name__)
    else:
        logger.info("reranker disabled (set RAG_RERANKER=lite to enable)")

    # Parent-chunk lookup — chunks the curated parquet again so we can hydrate
    # children → full parent text without storing parents in Qdrant payloads.
    enriched = Path("data/curated_enriched/jobs.parquet")
    base = Path("data/curated/jobs.parquet")
    parent_lookup = None
    try:
        parent_path = enriched if enriched.exists() else base
        parent_lookup = build_parent_lookup(parent_path)
        logger.info("parent lookup built :: %d parents", len(parent_lookup))
    except Exception as exc:  # noqa: BLE001
        logger.warning("parent lookup unavailable :: %s", exc)

    _retriever_singleton = HybridRetriever(
        qdrant_client=client,
        embedder=embedder,
        reranker=reranker,
        parent_lookup=parent_lookup,
    )
    return _retriever_singleton
