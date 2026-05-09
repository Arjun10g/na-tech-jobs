"""Chunk + embed + upsert all curated jobs into Qdrant.

Per CLAUDE.md §4 + §8 this is the **monthly** indexing pass after the
classifier enrichment lands. For dev iteration use ``--lite`` to swap
bge-m3 for MiniLM (dense-only, ~3 min on MPS); for production indexing
use the default bge-m3 (dense + sparse + optional multivec).

    # dev / fast iteration
    uv run python -m scripts.index_jobs --lite --limit 500

    # production index over the enriched curated parquet
    uv run python -m scripts.index_jobs --multivec

    # monitor a long run
    tail -f /tmp/index_jobs.log
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rag.chunking import chunk_jobs
from rag.embedder import (
    DEFAULT_DENSE_DIM,
    DEFAULT_LITE_DENSE_DIM,
    EmbeddingBatch,
    load_embedder,
)
from rag.qdrant_client import (
    COLLECTION_DENSE,
    COLLECTION_MULTIVEC,
    DEFAULT_LOCAL_PATH,
    collection_info,
    get_client,
    setup_dense_collection,
    setup_multivec_collection,
    upsert_dense,
    upsert_multivec,
)

logger = logging.getLogger("index_jobs")


def _select_curated_path(arg: str | None) -> Path:
    """Prefer the Phase 4 enriched parquet (carries versioned predictions)
    if present; fall back to the bare curated parquet otherwise."""
    if arg:
        return Path(arg)
    enriched = Path("data/curated_enriched/jobs.parquet")
    if enriched.exists():
        logger.info("using enriched parquet :: %s", enriched)
        return enriched
    return Path("data/curated/jobs.parquet")


def _embed_batch(embedder, texts: list[str], *, multivec: bool, batch_size: int):
    """Wrap embedder.encode so callers can stay batch-agnostic."""
    is_lite = embedder.__class__.__name__ == "_LiteEmbedder"
    return embedder.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=not is_lite,
        return_multivec=(not is_lite) and multivec,
    )


def index_jobs(
    *,
    curated_path: Path,
    qdrant_path: Path,
    lite: bool,
    multivec: bool,
    limit: int | None,
    embed_batch_size: int,
    upsert_batch_size: int,
    force_recreate: bool,
) -> dict:
    started = datetime.now(timezone.utc)

    df = pd.read_parquet(curated_path)
    if limit:
        df = df.head(limit).copy()
    logger.info("loaded %d jobs from %s", len(df), curated_path)

    rows = df.to_dict(orient="records")
    parents, children = chunk_jobs(rows)
    logger.info(
        "chunked :: %d parents, %d children (%.1f children/job)",
        len(parents),
        len(children),
        len(children) / max(len(rows), 1),
    )

    embedder = load_embedder(lite=lite)
    dense_dim = embedder.dense_dim
    logger.info(
        "embedder ready :: backend=%s dim=%d",
        embedder.__class__.__name__,
        dense_dim,
    )

    client = get_client(qdrant_path)
    setup_dense_collection(
        client,
        dense_dim=dense_dim,
        force_recreate=force_recreate,
    )
    multivec_to_index = multivec and not lite
    if multivec_to_index:
        setup_multivec_collection(
            client,
            multivec_dim=dense_dim,
            force_recreate=force_recreate,
        )

    n_dense = n_sparse = n_multivec = 0
    t0 = time.time()
    last_log = t0

    for batch_idx in range(0, len(children), embed_batch_size):
        batch = children[batch_idx : batch_idx + embed_batch_size]
        texts = [c.text for c in batch]
        out: EmbeddingBatch = _embed_batch(
            embedder,
            texts,
            multivec=multivec_to_index,
            batch_size=embed_batch_size,
        )
        n_dense += upsert_dense(
            client,
            batch,
            out.dense,
            out.sparse,
            batch_size=upsert_batch_size,
        )
        if out.sparse is not None:
            n_sparse += len(out.sparse)
        if multivec_to_index and out.multivec is not None:
            n_multivec += upsert_multivec(
                client,
                batch,
                out.multivec,
                batch_size=upsert_batch_size,
            )

        now = time.time()
        if now - last_log >= 30:
            done = batch_idx + len(batch)
            rate = done / max(now - t0, 1e-3)
            eta = (len(children) - done) / max(rate, 1e-3)
            logger.info(
                "progress :: %d/%d chunks (%.1f/s, eta %d min)",
                done,
                len(children),
                rate,
                int(eta // 60),
            )
            last_log = now

    finished = datetime.now(timezone.utc)
    info = collection_info(client, COLLECTION_DENSE)
    mv_info = collection_info(client, COLLECTION_MULTIVEC) if multivec_to_index else None

    summary = {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_sec": (finished - started).total_seconds(),
        "n_jobs": len(rows),
        "n_parents": len(parents),
        "n_children": len(children),
        "n_dense_upserted": n_dense,
        "n_sparse_upserted": n_sparse,
        "n_multivec_upserted": n_multivec,
        "embedder": embedder.__class__.__name__,
        "dense_dim": dense_dim,
        "qdrant_path": str(qdrant_path),
        "dense_collection_info": info,
        "multivec_collection_info": mv_info,
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--curated-path",
        default=None,
        help="Defaults to data/curated_enriched/jobs.parquet "
        "(falls back to data/curated/jobs.parquet)",
    )
    p.add_argument("--qdrant-path", default=str(DEFAULT_LOCAL_PATH))
    p.add_argument(
        "--lite",
        action="store_true",
        help="Use MiniLM (dense-only) instead of bge-m3 — for dev iteration",
    )
    p.add_argument(
        "--multivec",
        action="store_true",
        help="Also compute + index ColBERT multi-vectors (slow; needs bge-m3)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Cap rows for smoke-testing (e.g. --limit 100)"
    )
    p.add_argument("--embed-batch-size", type=int, default=8)
    p.add_argument("--upsert-batch-size", type=int, default=64)
    p.add_argument(
        "--force-recreate", action="store_true", help="Drop existing collections before re-indexing"
    )
    p.add_argument(
        "--out-summary",
        default="data/qdrant/index_summary.json",
        help="Where to write the JSON summary",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    summary = index_jobs(
        curated_path=_select_curated_path(args.curated_path),
        qdrant_path=Path(args.qdrant_path),
        lite=args.lite,
        multivec=args.multivec,
        limit=args.limit,
        embed_batch_size=args.embed_batch_size,
        upsert_batch_size=args.upsert_batch_size,
        force_recreate=args.force_recreate,
    )
    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote summary :: %s", out_path)
    print(json.dumps(summary, indent=2, default=str))

    # Sanity check the dim matches the embedder we picked.
    expected = DEFAULT_LITE_DENSE_DIM if args.lite else DEFAULT_DENSE_DIM
    assert summary["dense_dim"] == expected, (
        f"dim mismatch: indexed {summary['dense_dim']}, expected {expected}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
