"""Qdrant local-mode setup + upsert helpers for the RAG index.

Per CLAUDE.md §5 + §8 we use Qdrant in **local mode** with persistent
storage at ``data/qdrant/`` (matches the Spaces-Pro persistent disk
layout — same directory mounts at ``/data/qdrant`` in production).

Two collections:

- ``jobs_dense`` — child chunks with **named dense + sparse** vectors:
  - ``dense`` (cosine, HNSW + int8 scalar quantization)
  - ``sparse`` (BM25-style lexical; fed by bge-m3 lexical_weights)
- ``jobs_multivec`` — child chunks with **ColBERT-style multi-vector**
  (one vector per token, MaxSim comparator). Used for late-interaction
  reranking only on the top-K candidates returned from ``jobs_dense``.

Stable UUIDs derived from chunk_id let re-indexing be idempotent — the
same chunk maps to the same point id across runs.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from qdrant_client import QdrantClient

    from rag.chunking import ChildChunk

logger = logging.getLogger("rag.qdrant_client")

DEFAULT_LOCAL_PATH = Path("data/qdrant")
COLLECTION_DENSE = "jobs_dense"
COLLECTION_MULTIVEC = "jobs_multivec"

# Stable UUID namespace so the same chunk_id always produces the same
# point id, regardless of which machine indexes.
_NAMESPACE = uuid.UUID("4f3a9b32-1d2c-4f4e-8f2a-9b3c4d5e6f70")


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Stable UUID5 from the human-readable chunk id."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


# ── Client + setup ────────────────────────────────────────────────────────


def get_client(local_path: Path | str = DEFAULT_LOCAL_PATH) -> QdrantClient:
    """Return a local-mode QdrantClient pointed at ``local_path``."""
    from qdrant_client import QdrantClient

    p = Path(local_path)
    p.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(p))


def setup_dense_collection(
    client: QdrantClient,
    *,
    dense_dim: int,
    collection_name: str = COLLECTION_DENSE,
    force_recreate: bool = False,
    enable_quantization: bool = True,
) -> None:
    """Create the dense+sparse collection if it doesn't exist."""
    from qdrant_client import models

    if client.collection_exists(collection_name):
        if not force_recreate:
            logger.info("collection %s already exists — skipping", collection_name)
            return
        logger.warning("recreating collection %s", collection_name)
        client.delete_collection(collection_name)

    quantization = (
        models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            )
        )
        if enable_quantization
        else None
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
                quantization_config=quantization,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
            )
        },
    )
    logger.info(
        "created collection %s (dense_dim=%d, quantization=%s)",
        collection_name,
        dense_dim,
        "int8" if enable_quantization else "off",
    )


def setup_multivec_collection(
    client: QdrantClient,
    *,
    multivec_dim: int = 1024,
    collection_name: str = COLLECTION_MULTIVEC,
    force_recreate: bool = False,
) -> None:
    """Create the ColBERT-style multi-vector collection.

    ``multivec_dim`` is bge-m3's ColBERT output dimension (1024 by
    default; some checkpoints emit 128 — pass through from
    ``embedder.dense_dim`` or measure once at index time).
    """
    from qdrant_client import models

    if client.collection_exists(collection_name):
        if not force_recreate:
            logger.info("collection %s already exists — skipping", collection_name)
            return
        logger.warning("recreating collection %s", collection_name)
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "colbert": models.VectorParams(
                size=multivec_dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),  # disable HNSW; rerank-only
            )
        },
    )
    logger.info(
        "created collection %s (multivec_dim=%d, MaxSim)",
        collection_name,
        multivec_dim,
    )


# ── Upsert ────────────────────────────────────────────────────────────────


def sparse_to_qdrant(sparse: dict[int, float]):
    from qdrant_client import models

    return models.SparseVector(
        indices=list(sparse.keys()),
        values=list(sparse.values()),
    )


def upsert_dense(
    client: QdrantClient,
    chunks: list[ChildChunk],
    dense: np.ndarray,
    sparse: list[dict[int, float]] | None = None,
    *,
    collection_name: str = COLLECTION_DENSE,
    batch_size: int = 64,
) -> int:
    """Upsert one batch's worth of child chunks. Returns count written.

    ``dense.shape[0] == len(chunks)``. ``sparse`` (optional) is a list of
    ``{token_id: weight}`` dicts, same length.
    """
    from qdrant_client import models

    if dense.shape[0] != len(chunks):
        raise ValueError(f"dense rows ({dense.shape[0]}) != n_chunks ({len(chunks)})")
    if sparse is not None and len(sparse) != len(chunks):
        raise ValueError(f"sparse rows ({len(sparse)}) != n_chunks ({len(chunks)})")

    points: list[models.PointStruct] = []
    for i, chunk in enumerate(chunks):
        vector_payload: dict[str, Any] = {"dense": dense[i].tolist()}
        if sparse is not None:
            vector_payload["sparse"] = sparse_to_qdrant(sparse[i])
        payload = dict(chunk.payload)
        payload.update(
            {
                "chunk_id": chunk.child_chunk_id,
                "parent_chunk_id": chunk.parent_chunk_id,
                "job_id": chunk.job_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            }
        )
        points.append(
            models.PointStruct(
                id=chunk_id_to_point_id(chunk.child_chunk_id),
                vector=vector_payload,
                payload=payload,
            )
        )

    n_written = 0
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=False)
        n_written += len(batch)
    return n_written


def upsert_multivec(
    client: QdrantClient,
    chunks: list[ChildChunk],
    multivec: list[np.ndarray],
    *,
    collection_name: str = COLLECTION_MULTIVEC,
    batch_size: int = 32,
) -> int:
    """Upsert per-chunk ColBERT-style token vectors."""
    from qdrant_client import models

    if len(multivec) != len(chunks):
        raise ValueError(f"multivec rows ({len(multivec)}) != n_chunks ({len(chunks)})")

    points: list[models.PointStruct] = []
    for chunk, mv in zip(chunks, multivec, strict=True):
        # Qdrant expects list[list[float]] for multi-vector points.
        points.append(
            models.PointStruct(
                id=chunk_id_to_point_id(chunk.child_chunk_id),
                vector={"colbert": mv.tolist()},
                payload={
                    "chunk_id": chunk.child_chunk_id,
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "job_id": chunk.job_id,
                },
            )
        )

    n_written = 0
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=False)
        n_written += len(batch)
    return n_written


# ── Query helpers ─────────────────────────────────────────────────────────


def search_dense(
    client: QdrantClient,
    query_dense: np.ndarray,
    *,
    limit: int = 100,
    collection_name: str = COLLECTION_DENSE,
    qdrant_filter: Any | None = None,
):
    """Cosine search over the named ``dense`` vector."""
    return client.query_points(
        collection_name=collection_name,
        query=query_dense.tolist(),
        using="dense",
        limit=limit,
        with_payload=True,
        query_filter=qdrant_filter,
    ).points


def search_sparse(
    client: QdrantClient,
    query_sparse: dict[int, float],
    *,
    limit: int = 100,
    collection_name: str = COLLECTION_DENSE,
    qdrant_filter: Any | None = None,
):
    """BM25-style search over the ``sparse`` vector."""
    return client.query_points(
        collection_name=collection_name,
        query=sparse_to_qdrant(query_sparse),
        using="sparse",
        limit=limit,
        with_payload=True,
        query_filter=qdrant_filter,
    ).points


def collection_info(
    client: QdrantClient,
    collection_name: str = COLLECTION_DENSE,
) -> dict[str, Any]:
    """Light wrapper for the indexer summary."""
    info = client.get_collection(collection_name)
    return {
        "points_count": info.points_count,
        "vectors_count": getattr(info, "vectors_count", None),
        "status": str(info.status),
    }


def iter_in_chunks(it: Iterable, size: int):
    """Tiny utility — chunk an iterable for batched upsert."""
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
