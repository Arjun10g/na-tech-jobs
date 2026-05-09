"""Lazy singleton retriever for the Gradio app.

The first call loads:
- the Qdrant local-mode client (~10 ms). When the index isn't on disk
  (typical for a fresh Space cold start) we pull a pre-built tarball
  from the HF Dataset repo (``qdrant/qdrant_<encoder>_<version>.tar.gz``)
  and extract it in place; subsequent calls re-use the materialized
  directory.
- the embedder (MiniLM by default; swap to bge-m3 by setting
  ``RAG_EMBEDDER`` env var),
- optionally the cross-encoder reranker (``RAG_RERANKER`` env var to
  enable; defaults to OFF for cold-start latency),
- a parent-chunk lookup built from the enriched curated parquet.

After construction we cache and reuse for every query.

Env vars:
- ``RAG_QDRANT_PATH``       — local-mode Qdrant directory (default ``data/qdrant``).
- ``RAG_QDRANT_TARBALL``    — filename in the dataset repo to pull when
  the path is missing (default ``qdrant_minilm_v1.tar.gz``).
- ``HF_DATASET_REPO``       — the dataset repo to pull from
  (default ``arjun10g/na-tech-jobs``).
- ``RAG_EMBEDDER``          — embedder model id (default ``lite`` = MiniLM).
- ``RAG_RERANKER``          — ``off`` / ``lite`` / model id
  (default ``off``).
"""

from __future__ import annotations

import logging
import os
import tarfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("app.retriever_loader")

_retriever_singleton: Any | None = None
_qdrant_path: Path = Path("data/qdrant")
_DEFAULT_TARBALL = "qdrant_minilm_v1.tar.gz"


def _resolve_qdrant_path() -> Path:
    env = os.environ.get("RAG_QDRANT_PATH")
    return Path(env) if env else _qdrant_path


def _ensure_qdrant_index(qdrant_path: Path) -> None:
    """If the Qdrant directory isn't on disk, pull the tarball from the
    HF Dataset repo and extract.

    Fast-path: directory already populated (`collection/` subdir present).
    Slow-path: download tarball (~few hundred MB) → extract → done. This
    happens once per persistent-disk lifetime on the Space.
    """
    if (qdrant_path / "collection").exists():
        return  # Already materialized.

    tarball_name = os.environ.get("RAG_QDRANT_TARBALL", _DEFAULT_TARBALL)
    repo_id = os.environ.get("HF_DATASET_REPO", "arjun10g/na-tech-jobs")
    token = os.environ.get("HF_TOKEN")

    logger.info(
        "Qdrant index missing at %s — fetching %s from dataset:%s",
        qdrant_path,
        tarball_name,
        repo_id,
    )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(f"huggingface_hub not installed; can't fetch {tarball_name}") from exc

    try:
        local_tar = hf_hub_download(
            repo_id=repo_id,
            filename=f"qdrant/{tarball_name}",
            repo_type="dataset",
            token=token,
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Qdrant index missing locally and tarball "
            f"qdrant/{tarball_name} not in {repo_id} :: {exc}"
        ) from exc

    qdrant_path.mkdir(parents=True, exist_ok=True)
    logger.info("extracting %s → %s", local_tar, qdrant_path)
    with tarfile.open(local_tar, "r:*") as tf:
        # Tarball was created with `tar -cf - -C data qdrant`, so each
        # member starts with `qdrant/`. Strip that top-level prefix so
        # contents land directly in qdrant_path (which may have a
        # different name, e.g. `/data/qdrant` on the Space).
        members = []
        prefix = "qdrant/"
        for m in tf.getmembers():
            if m.name == "qdrant" or m.name == "qdrant/":
                continue
            if m.name.startswith(prefix):
                m.name = m.name[len(prefix) :]
            members.append(m)
        tf.extractall(path=qdrant_path, members=members)
    logger.info("Qdrant index ready :: %s", qdrant_path)


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
    _ensure_qdrant_index(qdrant_path)

    logger.info("opening qdrant local client at %s", qdrant_path)
    client = get_client(qdrant_path)
    if not client.collection_exists(COLLECTION_DENSE):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_DENSE}' does not exist at {qdrant_path}. "
            "Re-run `scripts.index_jobs` or check the published tarball."
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
