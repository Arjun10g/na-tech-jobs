"""bge-m3 embedder — one forward pass yields dense + sparse + multi-vector.

Per CLAUDE.md §5+§8 ``BAAI/bge-m3`` is the unified embedder powering the
Qdrant index *and* the salary-regressor description embedding. It returns
three outputs from one forward:

- **Dense**: 1024-dim float, mean-pooled. First-pass cosine retrieval.
- **Sparse**: lexical token-weight dict. BM25-style hybrid lexical search.
- **Multi-vector** (ColBERT-style): one ~128-dim vector per token, no
  pooling. Late-interaction reranking via MaxSim.

For development iteration we expose a lightweight encoder option
(``DEFAULT_LITE_MODEL_NAME``) — same MiniLM checkpoint used by the title
classifiers. It only produces dense embeddings and is *not* a
drop-in replacement for bge-m3 in production retrieval, but lets us
prototype the pipeline without 568 M params on every reload.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("rag.embedder")

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_LITE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DENSE_DIM = 1024  # bge-m3
DEFAULT_LITE_DENSE_DIM = 384  # MiniLM
DEFAULT_MAX_LENGTH = 512  # truncate before bge-m3 hits its 8k limit


@dataclass
class EmbeddingBatch:
    """Outputs from a batched encode."""

    dense: np.ndarray  # (n, dense_dim) float32, L2-normalized
    sparse: list[dict[int, float]] | None = None  # token-id → weight per row
    multivec: list[np.ndarray] | None = None  # per-row (n_tokens, multivec_dim)


# ── bge-m3 ────────────────────────────────────────────────────────────────


class _BGEM3Embedder:
    """Wrap ``FlagEmbedding.BGEM3FlagModel`` — the canonical bge-m3 backend."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        use_fp16: bool = True,
    ) -> None:
        import torch
        from FlagEmbedding import BGEM3FlagModel

        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        # bge-m3 fp16 is unstable on MPS; only enable on CUDA.
        effective_fp16 = use_fp16 and device == "cuda"
        logger.info(
            "loading %s on %s (fp16=%s, max_length=%d)",
            model_name,
            device,
            effective_fp16,
            max_length,
        )
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=effective_fp16,
            devices=[device] if device != "cpu" else None,
        )
        self.max_length = max_length
        self.dense_dim = DEFAULT_DENSE_DIM

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 8,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_multivec: bool = False,
    ) -> EmbeddingBatch:
        result: dict[str, Any] = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_multivec,
        )
        dense = (
            np.asarray(result.get("dense_vecs"), dtype=np.float32)
            if return_dense
            else np.zeros((len(texts), self.dense_dim), dtype=np.float32)
        )
        if return_dense and dense.size:
            # FlagEmbedding returns L2-normalized dense by default; assert.
            norms = np.linalg.norm(dense, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-2):
                dense = dense / np.clip(norms[:, None], 1e-9, None)

        sparse = None
        if return_sparse and result.get("lexical_weights"):
            # FlagEmbedding emits {str(token_id): float} — coerce keys.
            sparse = [
                {int(k): float(v) for k, v in row.items()} for row in result["lexical_weights"]
            ]
        multivec = None
        if return_multivec and result.get("colbert_vecs"):
            multivec = [np.asarray(v, dtype=np.float32) for v in result["colbert_vecs"]]
        return EmbeddingBatch(dense=dense, sparse=sparse, multivec=multivec)


# ── Lite (dev) encoder ────────────────────────────────────────────────────


class _LiteEmbedder:
    """Minimal sentence-transformer wrapper — dense only.

    Useful for fast local iteration on the chunking + retrieval pipeline
    without paying for bge-m3's load time on every dev reload. Production
    uses the bge-m3 backend.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LITE_MODEL_NAME,
        *,
        device: str | None = None,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        logger.info("loading lite encoder %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.dense_dim = DEFAULT_LITE_DENSE_DIM

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_multivec: bool = False,
    ) -> EmbeddingBatch:
        if return_sparse or return_multivec:
            logger.warning(
                "lite encoder ignores return_sparse / return_multivec — use bge-m3 for those modes",
            )
        dense = np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 500,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        return EmbeddingBatch(dense=dense)


# ── Public factory ────────────────────────────────────────────────────────


def load_embedder(
    model_name: str | None = None,
    *,
    lite: bool = False,
    **kwargs,
):
    """Return a configured embedder.

    - ``lite=True`` (or ``model_name="sentence-transformers/..."``):
      MiniLM dense-only fast path for dev iteration.
    - Otherwise: bge-m3 with dense + sparse + (opt) multivec.
    """
    if lite or (model_name and "MiniLM" in model_name):
        return _LiteEmbedder(model_name or DEFAULT_LITE_MODEL_NAME, **kwargs)
    return _BGEM3Embedder(model_name or DEFAULT_MODEL_NAME, **kwargs)
