"""Tests for rag.reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.reranker import (
    DEFAULT_LITE_MODEL_NAME,
    RerankResult,
    _BGERerankerV2M3,
    _LiteCrossEncoder,
    load_reranker,
    rerank,
)


@pytest.fixture
def fake_flag_reranker():
    with patch("FlagEmbedding.FlagReranker") as cls:
        instance = MagicMock()
        cls.return_value = instance

        def _score(pairs, batch_size=16):
            return [0.9 - 0.1 * i for i in range(len(pairs))]

        instance.compute_score.side_effect = _score
        yield cls


def test_bgereranker_score_returns_list_of_floats(fake_flag_reranker):
    r = _BGERerankerV2M3()
    out = r.score("query", ["a", "b", "c"])
    assert out == [0.9, 0.8, 0.7]


def test_bgereranker_score_handles_single_passage(fake_flag_reranker):
    r = _BGERerankerV2M3()
    out = r.score("query", ["only"])
    assert len(out) == 1
    assert isinstance(out[0], float)


def test_bgereranker_empty_passages_returns_empty(fake_flag_reranker):
    r = _BGERerankerV2M3()
    assert r.score("q", []) == []


@pytest.fixture
def fake_cross_encoder():
    with patch("sentence_transformers.CrossEncoder") as cls:
        instance = MagicMock()
        cls.return_value = instance
        instance.predict.return_value = np.array([0.5, 0.7, 0.2])
        yield cls


def test_lite_cross_encoder_score(fake_cross_encoder):
    r = _LiteCrossEncoder()
    out = r.score("q", ["a", "b", "c"])
    assert len(out) == 3
    assert out == [0.5, 0.7, 0.2]


def test_load_reranker_routes_lite_correctly():
    with patch("sentence_transformers.CrossEncoder") as _cls:
        r = load_reranker(lite=True)
        assert isinstance(r, _LiteCrossEncoder)


def test_load_reranker_routes_minilm_name_to_lite():
    with patch("sentence_transformers.CrossEncoder") as _cls:
        r = load_reranker(DEFAULT_LITE_MODEL_NAME)
        assert isinstance(r, _LiteCrossEncoder)


def test_load_reranker_returns_bge_by_default(fake_flag_reranker):
    r = load_reranker()
    assert isinstance(r, _BGERerankerV2M3)


def test_rerank_orders_by_score_descending():
    fake = MagicMock()
    fake.score.return_value = [0.3, 0.9, 0.5]
    out = rerank(fake, "q", ["a", "b", "c"])
    assert [r.index for r in out] == [1, 2, 0]
    assert out[0].score == 0.9
    assert all(isinstance(r, RerankResult) for r in out)


def test_rerank_respects_top_k():
    fake = MagicMock()
    fake.score.return_value = [0.3, 0.9, 0.5, 0.7, 0.1]
    out = rerank(fake, "q", ["a", "b", "c", "d", "e"], top_k=2)
    assert len(out) == 2
    assert [r.index for r in out] == [1, 3]


def test_rerank_empty_returns_empty():
    fake = MagicMock()
    fake.score.return_value = []
    assert rerank(fake, "q", []) == []
