"""Tests for eval.metrics — recall@k, MRR, nDCG@k, aggregate."""

from __future__ import annotations

import math

from eval.metrics import (
    aggregate,
    evaluate_query,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ── recall@k ──────────────────────────────────────────────────────────────


def test_recall_at_k_perfect():
    assert recall_at_k({"a", "b"}, ["a", "b", "c"], k=2) == 1.0
    assert recall_at_k({"a", "b"}, ["a", "b", "c"], k=10) == 1.0


def test_recall_at_k_partial():
    # 1 of 2 relevant in top-1 = 0.5
    assert recall_at_k({"a", "b"}, ["a", "x", "y"], k=1) == 0.5


def test_recall_at_k_zero_when_no_hits():
    assert recall_at_k({"a"}, ["x", "y", "z"], k=10) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k(set(), ["a", "b"], k=5) == 0.0


def test_recall_at_k_truncates_correctly():
    # b is at rank 3 — recall@2 misses it.
    assert recall_at_k({"a", "b"}, ["a", "x", "b"], k=2) == 0.5
    assert recall_at_k({"a", "b"}, ["a", "x", "b"], k=3) == 1.0


# ── MRR ───────────────────────────────────────────────────────────────────


def test_mrr_first_hit_at_rank_1():
    assert reciprocal_rank({"a"}, ["a", "b", "c"]) == 1.0


def test_mrr_first_hit_at_rank_3():
    assert reciprocal_rank({"a"}, ["x", "y", "a"]) == 1.0 / 3


def test_mrr_no_hit():
    assert reciprocal_rank({"a"}, ["x", "y", "z"]) == 0.0


def test_mrr_empty_relevant():
    assert reciprocal_rank(set(), ["a", "b"]) == 0.0


def test_mrr_uses_first_hit_only():
    # Even with 2 relevant docs, MRR cares only about the first found.
    assert reciprocal_rank({"a", "b"}, ["x", "a", "b"]) == 0.5


# ── nDCG ──────────────────────────────────────────────────────────────────


def test_ndcg_perfect_ranking():
    # All relevant at top → nDCG = 1.0
    assert ndcg_at_k({"a", "b"}, ["a", "b", "c"], k=10) == 1.0


def test_ndcg_zero_when_nothing_retrieved_in_topk():
    assert ndcg_at_k({"a"}, ["x", "y", "z"], k=2) == 0.0


def test_ndcg_decreases_with_rank():
    # Same hit, deeper position → lower nDCG.
    n_top = ndcg_at_k({"a"}, ["a", "x", "y"], k=10)
    n_bot = ndcg_at_k({"a"}, ["x", "y", "a"], k=10)
    assert n_top > n_bot
    # rank 1 → DCG=1, IDCG=1, nDCG=1
    assert math.isclose(n_top, 1.0)


def test_ndcg_with_partial_hits():
    # 2 relevant total, 1 hit at rank 2:
    #   DCG = 1/log2(3) ≈ 0.6309
    #   IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    #   nDCG ≈ 0.387
    val = ndcg_at_k({"a", "b"}, ["x", "a", "y"], k=10)
    assert 0.38 < val < 0.40


def test_ndcg_empty_relevant():
    assert ndcg_at_k(set(), ["a"], k=5) == 0.0


# ── evaluate_query ────────────────────────────────────────────────────────


def test_evaluate_query_returns_all_metrics():
    out = evaluate_query(
        query_id="q1",
        query="hello",
        relevant_ids={"a", "b"},
        retrieved_ids=["a", "x", "b", "y", "z"],
    )
    assert out["query_id"] == "q1"
    assert out["query"] == "hello"
    assert out["n_relevant"] == 2
    assert out["n_retrieved"] == 5
    assert out["recall_at_5"] == 1.0
    assert out["recall_at_10"] == 1.0
    assert out["recall_at_20"] == 1.0
    assert out["mrr"] == 1.0
    assert 0.0 < out["ndcg_at_10"] <= 1.0


# ── aggregate ─────────────────────────────────────────────────────────────


def test_aggregate_excludes_zero_relevant_queries():
    rows = [
        evaluate_query(
            query_id="q1",
            query="x",
            relevant_ids={"a"},
            retrieved_ids=["a"],
        ),
        evaluate_query(
            query_id="q2",
            query="y",
            relevant_ids=set(),
            retrieved_ids=["b"],  # excluded
        ),
    ]
    agg = aggregate(rows)
    assert agg["n_queries"] == 1
    assert agg["recall_at_5"] == 1.0


def test_aggregate_empty_returns_zeros():
    agg = aggregate([])
    assert agg["n_queries"] == 0
    assert agg["recall_at_5"] == 0.0
    assert agg["mrr"] == 0.0


def test_aggregate_means_correctly():
    rows = [
        evaluate_query(query_id="q1", query="x", relevant_ids={"a"}, retrieved_ids=["a"]),  # mrr=1
        evaluate_query(
            query_id="q2", query="y", relevant_ids={"b"}, retrieved_ids=["x", "b"]
        ),  # mrr=0.5
    ]
    agg = aggregate(rows)
    assert agg["n_queries"] == 2
    assert math.isclose(agg["mrr"], 0.75)
