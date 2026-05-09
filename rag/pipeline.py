"""Hybrid retrieval pipeline: query → embed → first-pass → rerank → hydrate.

Orchestrates the flow from CLAUDE.md §8:

1. **First-pass hybrid** over ``jobs_dense``:
   - Dense cosine search (top 100)
   - Sparse BM25 search (top 100, when the embedder produces sparse)
   - Reciprocal Rank Fusion (RRF, k=60)
2. **Cross-encoder rerank** on the top 100 → keep top 20.
3. **Parent-chunk hydration**: dedupe child chunks → unique parents,
   keep top 5-10 parents.
4. (Optional) ColBERT MaxSim — deferred to v1.1 when multivec is indexed.

Every step is independent + tolerant of missing data:
- If sparse vectors aren't in the index, fall back to dense-only.
- If reranker fails to load, skip rerank step (return first-pass).
- If parents aren't loadable from the curated parquet, fall back to
  the indexed chunk text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger("rag.pipeline")


# ── Result types ──────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    chunk_id: str
    parent_chunk_id: str
    job_id: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    score_dense: float | None = None
    score_sparse: float | None = None
    score_rrf: float | None = None
    score_rerank: float | None = None


@dataclass
class RetrievedJob:
    job_id: str
    parent_chunk_id: str
    text: str  # parent chunk content (LLM-ready context)
    payload: dict[str, Any]
    score: float  # final ranking score (rerank if available, else RRF)
    contributing_child_ids: list[str] = field(default_factory=list)


# ── RRF ───────────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    rank_lists: list[list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """RRF score per id across multiple ranked lists.

    `score(id) = sum_i 1/(k + rank_i(id))` with rank starting at 1.
    Higher = better. ``k=60`` is the standard from the original RRF paper.
    """
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, item in enumerate(ranks, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


# ── Filter helpers ────────────────────────────────────────────────────────


def build_filter(
    *,
    countries: list[str] | None = None,
    seniority_labels: list[str] | None = None,
    role_families: list[str] | None = None,
    min_predicted_salary_usd: float | None = None,
    max_predicted_salary_usd: float | None = None,
    posted_after: str | None = None,  # ISO timestamp
):
    """Translate UI filters → Qdrant Filter."""
    from qdrant_client import models

    must: list[Any] = []
    if countries:
        must.append(models.FieldCondition(key="country", match=models.MatchAny(any=countries)))
    if seniority_labels:
        must.append(
            models.FieldCondition(
                key="seniority_label_v1", match=models.MatchAny(any=seniority_labels)
            )
        )
    if role_families:
        must.append(
            models.FieldCondition(key="role_family_v1", match=models.MatchAny(any=role_families))
        )
    if min_predicted_salary_usd is not None or max_predicted_salary_usd is not None:
        must.append(
            models.FieldCondition(
                key="predicted_salary_usd_v1",
                range=models.Range(
                    gte=min_predicted_salary_usd,
                    lte=max_predicted_salary_usd,
                ),
            )
        )
    if posted_after:
        must.append(
            models.FieldCondition(
                key="posted_at",
                range=models.DatetimeRange(gte=posted_after),
            )
        )
    return models.Filter(must=must) if must else None


# ── The pipeline ──────────────────────────────────────────────────────────


@dataclass
class HybridRetriever:
    """End-to-end retrieval over a populated Qdrant index.

    Construct once at app startup; reuse for every query.
    """

    qdrant_client: QdrantClient
    embedder: Any  # rag.embedder._BGEM3Embedder | _LiteEmbedder
    reranker: Any | None = None  # rag.reranker reranker (optional)
    parent_lookup: dict[str, dict[str, Any]] | None = None  # parent_chunk_id → {text, payload}

    first_pass_dense_k: int = 100
    first_pass_sparse_k: int = 100
    rerank_k: int = 20
    final_top_k: int = 10
    rrf_k: int = 60

    # ── First pass ────────────────────────────────────────────────────────

    def _search_dense(self, query: str, *, limit: int, qdrant_filter):
        from rag.qdrant_client import COLLECTION_DENSE

        out = self.embedder.encode([query], batch_size=1)
        return self.qdrant_client.search(
            collection_name=COLLECTION_DENSE,
            query_vector=("dense", out.dense[0].tolist()),
            limit=limit,
            with_payload=True,
            query_filter=qdrant_filter,
        ), out

    def _search_sparse(self, sparse_query: dict[int, float], *, limit: int, qdrant_filter):
        from qdrant_client import models

        from rag.qdrant_client import COLLECTION_DENSE, sparse_to_qdrant

        return self.qdrant_client.search(
            collection_name=COLLECTION_DENSE,
            query_vector=models.NamedSparseVector(
                name="sparse",
                vector=sparse_to_qdrant(sparse_query),
            ),
            limit=limit,
            with_payload=True,
            query_filter=qdrant_filter,
        )

    def first_pass(
        self,
        query: str,
        *,
        qdrant_filter: Any | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid dense+sparse retrieval with RRF fusion. Falls back to
        dense-only when the index has no sparse vectors (e.g., MiniLM-only)."""
        dense_hits, embed_out = self._search_dense(
            query,
            limit=self.first_pass_dense_k,
            qdrant_filter=qdrant_filter,
        )
        dense_ids = [str(h.id) for h in dense_hits]
        dense_payload = {str(h.id): (h.payload or {}, float(h.score)) for h in dense_hits}

        sparse_payload: dict[str, tuple[dict, float]] = {}
        sparse_ids: list[str] = []
        if embed_out.sparse:
            try:
                sparse_hits = self._search_sparse(
                    embed_out.sparse[0],
                    limit=self.first_pass_sparse_k,
                    qdrant_filter=qdrant_filter,
                )
                sparse_ids = [str(h.id) for h in sparse_hits]
                sparse_payload = {str(h.id): (h.payload or {}, float(h.score)) for h in sparse_hits}
            except Exception as exc:  # noqa: BLE001
                logger.warning("sparse search failed; falling back to dense-only :: %s", exc)

        if sparse_ids:
            rrf = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.rrf_k)
        else:
            rrf = reciprocal_rank_fusion([dense_ids], k=self.rrf_k)

        ranked_ids = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[
            : self.first_pass_dense_k
        ]

        out: list[RetrievedChunk] = []
        for pid in ranked_ids:
            payload, dense_score = dense_payload.get(pid, ({}, None))
            if not payload:
                payload, _ = sparse_payload.get(pid, ({}, None))
            sparse_score = sparse_payload.get(pid, ({}, None))[1] if sparse_payload else None
            out.append(
                RetrievedChunk(
                    chunk_id=payload.get("chunk_id", pid),
                    parent_chunk_id=payload.get("parent_chunk_id", ""),
                    job_id=payload.get("job_id", ""),
                    text=payload.get("text", ""),
                    payload={k: v for k, v in payload.items() if k != "text"},
                    score_dense=dense_score,
                    score_sparse=sparse_score,
                    score_rrf=rrf[pid],
                )
            )
        return out

    # ── Rerank ────────────────────────────────────────────────────────────

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self.reranker is None or not candidates:
            return candidates[: self.rerank_k]
        from rag.reranker import rerank as _rerank

        passages = [c.text for c in candidates]
        ranked = _rerank(self.reranker, query, passages, top_k=self.rerank_k)
        out: list[RetrievedChunk] = []
        for r in ranked:
            c = candidates[r.index]
            c.score_rerank = r.score
            out.append(c)
        return out

    # ── Hydrate to parents ────────────────────────────────────────────────

    def hydrate_parents(self, children: list[RetrievedChunk]) -> list[RetrievedJob]:
        """Dedupe children by parent_chunk_id, keep highest-scoring child per
        parent, return parents in score order."""
        by_parent: dict[str, RetrievedChunk] = {}
        contributors: dict[str, list[str]] = {}
        for c in children:
            pid = c.parent_chunk_id or c.chunk_id
            contributors.setdefault(pid, []).append(c.chunk_id)
            best = by_parent.get(pid)
            if best is None or self._effective_score(c) > self._effective_score(best):
                by_parent[pid] = c

        results: list[RetrievedJob] = []
        for pid, child in by_parent.items():
            parent_text = child.text  # default to child's text
            parent_payload = dict(child.payload)
            if self.parent_lookup and pid in self.parent_lookup:
                parent = self.parent_lookup[pid]
                parent_text = parent.get("text", parent_text)
                parent_payload = parent.get("payload", parent_payload)
            results.append(
                RetrievedJob(
                    job_id=child.job_id,
                    parent_chunk_id=pid,
                    text=parent_text,
                    payload=parent_payload,
                    score=self._effective_score(child),
                    contributing_child_ids=contributors[pid],
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: self.final_top_k]

    @staticmethod
    def _effective_score(c: RetrievedChunk) -> float:
        if c.score_rerank is not None:
            return c.score_rerank
        if c.score_rrf is not None:
            return c.score_rrf
        return c.score_dense or 0.0

    # ── End-to-end ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        qdrant_filter: Any | None = None,
    ) -> list[RetrievedJob]:
        """The full retrieval flow."""
        children = self.first_pass(query, qdrant_filter=qdrant_filter)
        if not children:
            return []
        children = self.rerank(query, children)
        return self.hydrate_parents(children)


# ── Parent-chunk lookup builder ───────────────────────────────────────────


def build_parent_lookup(
    curated_path: Path = Path("data/curated_enriched/jobs.parquet"),
) -> dict[str, dict[str, Any]]:
    """Re-chunk the curated parquet into parents and build a lookup so the
    pipeline can hydrate children → full parent text at query time.

    This is faster than storing parent text in Qdrant payloads (which
    would balloon the index) and fast enough to call once at app startup.
    """
    import pandas as pd

    from rag.chunking import chunk_jobs

    df = pd.read_parquet(curated_path)
    rows = df.to_dict(orient="records")
    parents, _children = chunk_jobs(rows)
    return {
        p.parent_chunk_id: {"text": p.text, "payload": p.payload, "job_id": p.job_id}
        for p in parents
    }
