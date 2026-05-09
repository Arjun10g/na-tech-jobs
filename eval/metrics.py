"""Retrieval metrics — recall@k, MRR, nDCG@k.

Per CLAUDE.md §8 the eval harness measures every retrieval variant on the
same labeled query set so we can compare apples-to-apples in the README's
eval table. All metrics here take:

- ``relevant_ids``: set of gold-relevant job ids for one query
- ``retrieved_ids``: ordered list of job ids the retriever returned (top
  result first)

We aggregate **per job_id**, not per chunk_id, because the eval question is
"did the retriever surface the right *job*?" — a job's parent and child
chunks all collapse to the same answer.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypedDict


class QueryEval(TypedDict):
    """One row of the eval harness output."""

    query_id: str
    query: str
    n_relevant: int
    n_retrieved: int
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float


def _truncate(retrieved: Iterable[str], k: int) -> list[str]:
    out: list[str] = []
    for r in retrieved:
        if len(out) >= k:
            break
        out.append(r)
    return out


def recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """``|relevant ∩ top-k| / |relevant|``."""
    if not relevant_ids:
        return 0.0
    top = set(_truncate(retrieved_ids, k))
    hits = top & relevant_ids
    return len(hits) / len(relevant_ids)


def reciprocal_rank(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    """Mean Reciprocal Rank for one query — 1/rank of the first relevant
    hit (1-indexed), or 0 if no relevant doc was retrieved."""
    if not relevant_ids:
        return 0.0
    for i, r in enumerate(retrieved_ids, start=1):
        if r in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """nDCG@k with binary relevance.

    DCG = sum over rank i of rel_i / log2(i+1).
    IDCG = perfect ranking — all relevant docs at the top.
    """
    if not relevant_ids:
        return 0.0
    top = _truncate(retrieved_ids, k)
    dcg = 0.0
    for i, r in enumerate(top, start=1):
        if r in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    n_perfect = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_perfect + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(
    *,
    query_id: str,
    query: str,
    relevant_ids: set[str],
    retrieved_ids: list[str],
) -> QueryEval:
    """Compute all metrics for one query."""
    return {
        "query_id": query_id,
        "query": query,
        "n_relevant": len(relevant_ids),
        "n_retrieved": len(retrieved_ids),
        "recall_at_5": recall_at_k(relevant_ids, retrieved_ids, 5),
        "recall_at_10": recall_at_k(relevant_ids, retrieved_ids, 10),
        "recall_at_20": recall_at_k(relevant_ids, retrieved_ids, 20),
        "mrr": reciprocal_rank(relevant_ids, retrieved_ids),
        "ndcg_at_10": ndcg_at_k(relevant_ids, retrieved_ids, 10),
    }


def aggregate(per_query: list[QueryEval]) -> dict[str, float]:
    """Mean of each metric across queries.

    Queries with empty ``relevant_ids`` are excluded — including them
    would artificially deflate the mean (their per-query metric is 0).
    """
    eligible = [q for q in per_query if q["n_relevant"] > 0]
    n = len(eligible)
    if n == 0:
        return {
            "n_queries": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "mrr": 0.0,
            "ndcg_at_10": 0.0,
        }
    return {
        "n_queries": n,
        "recall_at_5": sum(q["recall_at_5"] for q in eligible) / n,
        "recall_at_10": sum(q["recall_at_10"] for q in eligible) / n,
        "recall_at_20": sum(q["recall_at_20"] for q in eligible) / n,
        "mrr": sum(q["mrr"] for q in eligible) / n,
        "ndcg_at_10": sum(q["ndcg_at_10"] for q in eligible) / n,
    }
