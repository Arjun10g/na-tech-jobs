"""Tests for rag.pipeline — RRF, hydration, end-to-end orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np

from rag.embedder import EmbeddingBatch
from rag.pipeline import (
    HybridRetriever,
    RetrievedChunk,
    build_filter,
    reciprocal_rank_fusion,
)

# ── RRF ───────────────────────────────────────────────────────────────────


def test_rrf_single_list_returns_descending_scores():
    scores = reciprocal_rank_fusion([["a", "b", "c"]])
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_two_lists_combines():
    # Item "x" in both lists at top should beat items in only one.
    scores = reciprocal_rank_fusion([["x", "y", "z"], ["x", "w"]])
    assert scores["x"] > scores["y"]
    assert scores["x"] > scores["w"]


def test_rrf_k_parameter_changes_score():
    s_k60 = reciprocal_rank_fusion([["a"]], k=60)
    s_k10 = reciprocal_rank_fusion([["a"]], k=10)
    assert s_k10["a"] > s_k60["a"]


def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[]]) == {}


# ── Filter builder ────────────────────────────────────────────────────────


def test_build_filter_no_args_returns_none():
    assert build_filter() is None


def test_build_filter_country_match_any():
    f = build_filter(countries=["US", "CA"])
    assert f is not None
    assert len(f.must) == 1


def test_build_filter_combines_multiple_conditions():
    f = build_filter(
        countries=["US"],
        seniority_labels=["senior", "staff"],
        min_predicted_salary_usd=150_000,
    )
    assert f is not None
    assert len(f.must) == 3


def test_build_filter_salary_range_only_min():
    f = build_filter(min_predicted_salary_usd=100_000)
    assert f is not None
    assert len(f.must) == 1


# ── Hydration ─────────────────────────────────────────────────────────────


def _chunk(chunk_id, parent_chunk_id, job_id, score_rerank=None, text="content"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        parent_chunk_id=parent_chunk_id,
        job_id=job_id,
        text=text,
        payload={"id": job_id, "title": f"Title {job_id}"},
        score_dense=0.5,
        score_rrf=0.1,
        score_rerank=score_rerank,
    )


def _make_retriever(parent_lookup=None, reranker=None):
    return HybridRetriever(
        qdrant_client=MagicMock(),
        embedder=MagicMock(),
        reranker=reranker,
        parent_lookup=parent_lookup,
        first_pass_dense_k=100,
        rerank_k=20,
        final_top_k=10,
    )


def test_hydrate_dedupes_children_to_unique_parents():
    children = [
        _chunk("a::c0", "a::p0", "a", score_rerank=0.9),
        _chunk("a::c1", "a::p0", "a", score_rerank=0.8),  # same parent
        _chunk("b::c0", "b::p0", "b", score_rerank=0.7),
    ]
    retr = _make_retriever()
    out = retr.hydrate_parents(children)
    assert len(out) == 2  # a::p0 and b::p0
    parent_ids = [r.parent_chunk_id for r in out]
    assert "a::p0" in parent_ids
    assert "b::p0" in parent_ids


def test_hydrate_keeps_highest_score_among_same_parent():
    children = [
        _chunk("a::c0", "a::p0", "a", score_rerank=0.5),
        _chunk("a::c1", "a::p0", "a", score_rerank=0.95),
    ]
    retr = _make_retriever()
    out = retr.hydrate_parents(children)
    assert len(out) == 1
    assert out[0].score == 0.95
    # Both children listed.
    assert set(out[0].contributing_child_ids) == {"a::c0", "a::c1"}


def test_hydrate_orders_by_score_desc():
    children = [
        _chunk("a::c0", "a::p0", "a", score_rerank=0.3),
        _chunk("b::c0", "b::p0", "b", score_rerank=0.9),
        _chunk("c::c0", "c::p0", "c", score_rerank=0.6),
    ]
    retr = _make_retriever()
    out = retr.hydrate_parents(children)
    assert [r.job_id for r in out] == ["b", "c", "a"]


def test_hydrate_uses_parent_lookup_when_provided():
    parent_lookup = {"a::p0": {"text": "FULL PARENT TEXT", "payload": {"id": "a", "title": "X"}}}
    children = [_chunk("a::c0", "a::p0", "a", score_rerank=0.9, text="just child")]
    retr = _make_retriever(parent_lookup=parent_lookup)
    out = retr.hydrate_parents(children)
    assert out[0].text == "FULL PARENT TEXT"


def test_hydrate_falls_back_to_child_text_when_no_lookup():
    children = [_chunk("a::c0", "a::p0", "a", score_rerank=0.9, text="child only")]
    retr = _make_retriever()
    out = retr.hydrate_parents(children)
    assert out[0].text == "child only"


def test_hydrate_respects_final_top_k():
    children = [
        _chunk(f"j{i}::c0", f"j{i}::p0", f"j{i}", score_rerank=0.5 + 0.01 * i) for i in range(20)
    ]
    retr = HybridRetriever(qdrant_client=MagicMock(), embedder=MagicMock(), final_top_k=5)
    out = retr.hydrate_parents(children)
    assert len(out) == 5


def test_hydrate_score_priority_rerank_over_rrf_over_dense():
    # No rerank → use RRF.
    c1 = RetrievedChunk(
        chunk_id="x::c0",
        parent_chunk_id="x::p0",
        job_id="x",
        text="t",
        score_dense=0.1,
        score_rrf=0.5,
        score_rerank=None,
    )
    retr = _make_retriever()
    out = retr.hydrate_parents([c1])
    assert out[0].score == 0.5

    # No rerank, no rrf → use dense.
    c2 = RetrievedChunk(
        chunk_id="x::c0",
        parent_chunk_id="x::p0",
        job_id="x",
        text="t",
        score_dense=0.1,
        score_rrf=None,
        score_rerank=None,
    )
    out = retr.hydrate_parents([c2])
    assert out[0].score == 0.1


# ── Rerank step (mocked) ──────────────────────────────────────────────────


def test_rerank_step_returns_first_pass_when_no_reranker():
    children = [
        _chunk("a::c0", "a::p0", "a"),
        _chunk("b::c0", "b::p0", "b"),
    ]
    retr = _make_retriever(reranker=None)
    out = retr.rerank("q", children)
    # Without a reranker we just truncate to rerank_k.
    assert out == children


def test_rerank_step_attaches_scores_when_reranker_present():
    children = [
        _chunk("a::c0", "a::p0", "a"),
        _chunk("b::c0", "b::p0", "b"),
    ]
    fake = MagicMock()
    fake.score.return_value = [0.2, 0.95]
    retr = _make_retriever(reranker=fake)
    out = retr.rerank("q", children)
    # Reranker reverses — b ahead of a.
    assert out[0].chunk_id == "b::c0"
    assert out[0].score_rerank == 0.95
    assert out[1].score_rerank == 0.2


# ── End-to-end (with mocked qdrant) ───────────────────────────────────────


@dataclass
class _FakeHit:
    id: str
    score: float
    payload: dict


def test_search_end_to_end_dense_only_path():
    qclient = MagicMock()
    qclient.search.return_value = [
        _FakeHit(
            id="00000000-0000-0000-0000-000000000001",
            score=0.9,
            payload={
                "chunk_id": "a::c0",
                "parent_chunk_id": "a::p0",
                "job_id": "a",
                "text": "alpha text",
                "title": "A",
            },
        ),
        _FakeHit(
            id="00000000-0000-0000-0000-000000000002",
            score=0.5,
            payload={
                "chunk_id": "b::c0",
                "parent_chunk_id": "b::p0",
                "job_id": "b",
                "text": "beta text",
                "title": "B",
            },
        ),
    ]

    embedder = MagicMock()
    embedder.encode.return_value = EmbeddingBatch(
        dense=np.zeros((1, 384), dtype=np.float32),
        sparse=None,
    )

    retr = HybridRetriever(qdrant_client=qclient, embedder=embedder)
    results = retr.search("test query")
    assert len(results) == 2
    assert results[0].job_id == "a"  # higher RRF rank
    assert results[0].score is not None
