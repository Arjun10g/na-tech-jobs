"""Cross-encoder reranking for top-K retrieved candidates.

Per CLAUDE.md §5+§8 the reranker is ``BAAI/bge-reranker-v2-m3`` — a
multilingual cross-encoder that scores (query, passage) pairs jointly.
Cross-encoders are slow but accurate: we run them only on the ~100
candidates returned from the first-pass hybrid retrieval, then keep
the top 20.

For dev iteration we expose a lite path that uses
``cross-encoder/ms-marco-MiniLM-L-6-v2`` (22 M params) — a 10x smaller
model that ships fast and runs fine on CPU. It's noticeably weaker than
bge-reranker-v2-m3 on multilingual content but adequate for local
prototyping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("rag.reranker")

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_LITE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankResult:
    index: int  # original index in the input list
    score: float


class _BGERerankerV2M3:
    """Wrap ``FlagEmbedding.FlagReranker`` for bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        use_fp16: bool = True,
    ) -> None:
        import torch
        from FlagEmbedding import FlagReranker

        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        effective_fp16 = use_fp16 and device == "cuda"
        logger.info("loading reranker %s on %s (fp16=%s)", model_name, device, effective_fp16)
        self.model = FlagReranker(model_name, use_fp16=effective_fp16)

    def score(self, query: str, passages: list[str], *, batch_size: int = 16) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self.model.compute_score(pairs, batch_size=batch_size)
        # FlagReranker returns scalar for n=1, list for n>=2.
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        return [float(s) for s in scores]


class _LiteCrossEncoder:
    """Tiny ms-marco MiniLM cross-encoder for dev iteration."""

    def __init__(
        self,
        model_name: str = DEFAULT_LITE_MODEL_NAME,
        *,
        device: str | None = None,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        logger.info("loading lite cross-encoder %s on %s", model_name, device)
        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: list[str], *, batch_size: int = 32) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return [float(s) for s in np.asarray(scores).reshape(-1)]


# ── Public API ────────────────────────────────────────────────────────────


def load_reranker(
    model_name: str | None = None,
    *,
    lite: bool = False,
    **kwargs: Any,
):
    """Return a configured cross-encoder reranker.

    - ``lite=True`` (or any name containing ``MiniLM``):
      ms-marco MiniLM cross-encoder.
    - Otherwise: bge-reranker-v2-m3.
    """
    if lite or (model_name and "MiniLM" in model_name):
        return _LiteCrossEncoder(model_name or DEFAULT_LITE_MODEL_NAME, **kwargs)
    return _BGERerankerV2M3(model_name or DEFAULT_MODEL_NAME, **kwargs)


def rerank(
    reranker,
    query: str,
    passages: list[str],
    *,
    top_k: int | None = None,
    batch_size: int = 16,
) -> list[RerankResult]:
    """Score every passage, return them sorted high-to-low.

    The returned ``index`` field is the position in the input ``passages``
    list — callers join back to their original metadata.
    """
    scores = reranker.score(query, passages, batch_size=batch_size)
    ranked = [
        RerankResult(index=i, score=s)
        for i, s in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ]
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked
