"""Tests for rag.embedder.

The bge-m3 backend is heavy (568M params) — we test it with a fake to
keep CI fast. Real-model integration is exercised via the indexing
script. The lite encoder is tested end-to-end since MiniLM is small
(22M) and already cached for the title classifiers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.embedder import (
    DEFAULT_DENSE_DIM,
    DEFAULT_LITE_DENSE_DIM,
    EmbeddingBatch,
    _BGEM3Embedder,
    _LiteEmbedder,
    load_embedder,
)

# ── bge-m3 (mocked FlagEmbedding) ─────────────────────────────────────────


@pytest.fixture
def fake_bgem3():
    """Patch FlagEmbedding.BGEM3FlagModel with a lightweight stand-in."""
    with patch("FlagEmbedding.BGEM3FlagModel") as cls:
        instance = MagicMock()
        cls.return_value = instance

        def _encode(texts, **kw):
            n = len(texts)
            dense = (
                np.random.default_rng(0).standard_normal((n, DEFAULT_DENSE_DIM)).astype(np.float32)
            )
            dense /= np.linalg.norm(dense, axis=1, keepdims=True)
            out = {"dense_vecs": dense}
            if kw.get("return_sparse"):
                out["lexical_weights"] = [{"42": 0.7, "100": 0.3} for _ in range(n)]
            if kw.get("return_colbert_vecs"):
                # 8 tokens × 128 dim per row.
                out["colbert_vecs"] = [
                    np.random.default_rng(i).standard_normal((8, 128)).astype(np.float32)
                    for i in range(n)
                ]
            return out

        instance.encode.side_effect = _encode
        yield cls


def test_bgem3_encoder_returns_dense_only(fake_bgem3):
    emb = _BGEM3Embedder()
    out = emb.encode(["a", "b", "c"], return_sparse=False, return_multivec=False)
    assert isinstance(out, EmbeddingBatch)
    assert out.dense.shape == (3, DEFAULT_DENSE_DIM)
    assert np.allclose(np.linalg.norm(out.dense, axis=1), 1.0, atol=1e-3)
    assert out.sparse is None
    assert out.multivec is None


def test_bgem3_encoder_returns_sparse_when_requested(fake_bgem3):
    emb = _BGEM3Embedder()
    out = emb.encode(["a"], return_sparse=True)
    assert out.sparse is not None
    assert len(out.sparse) == 1
    # Token ids should be coerced from str → int.
    assert all(isinstance(k, int) for k in out.sparse[0])


def test_bgem3_encoder_returns_multivec_when_requested(fake_bgem3):
    emb = _BGEM3Embedder()
    out = emb.encode(["a", "b"], return_multivec=True)
    assert out.multivec is not None
    assert len(out.multivec) == 2
    # Each row is (n_tokens, dim).
    assert out.multivec[0].ndim == 2
    assert out.multivec[0].shape[1] == 128


# ── Lite encoder (real MiniLM) ────────────────────────────────────────────


def test_lite_encoder_dense_shape_and_normalization():
    emb = _LiteEmbedder()
    out = emb.encode(["hello world", "another sentence"])
    assert out.dense.shape == (2, DEFAULT_LITE_DENSE_DIM)
    assert np.allclose(np.linalg.norm(out.dense, axis=1), 1.0, atol=1e-3)
    assert out.sparse is None
    assert out.multivec is None


def test_lite_encoder_warns_on_unsupported_modes(caplog):
    emb = _LiteEmbedder()
    with caplog.at_level("WARNING", logger="rag.embedder"):
        emb.encode(["x"], return_sparse=True, return_multivec=True)
    assert "lite encoder ignores" in caplog.text


# ── Factory ───────────────────────────────────────────────────────────────


def test_load_embedder_returns_lite_when_flagged():
    emb = load_embedder(lite=True)
    assert isinstance(emb, _LiteEmbedder)


def test_load_embedder_routes_minilm_name_to_lite():
    emb = load_embedder("sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(emb, _LiteEmbedder)


def test_load_embedder_returns_bgem3_when_default(fake_bgem3):
    emb = load_embedder()
    assert isinstance(emb, _BGEM3Embedder)
