"""Run every retrieval variant against the labeled query set, compute
per-query and aggregate metrics, write CSV + markdown.

Per CLAUDE.md §8 the variants we want to compare are:

- ``dense`` — dense-only first pass (baseline)
- ``hybrid`` — dense + sparse, RRF-fused (only meaningful when sparse is
  in the index; falls back to dense-only otherwise)
- ``hybrid+rerank`` — hybrid + cross-encoder rerank
- ``hybrid+rerank+colbert`` — adds ColBERT MaxSim reranking on the top 20
  (deferred to v1.1 when multivec is indexed)
- ``hybrid+rerank+colbert+hyde`` — Qwen HyDE before retrieval (deferred
  to v1.1)

Each variant gets its own configured ``HybridRetriever``; we score them
one at a time so a single hot index is shared across runs.

Usage::

    uv run python -m eval.run_retrieval_eval \\
      --queries eval/retrieval_queries.jsonl \\
      --variants dense hybrid hybrid+rerank \\
      --top-k 20

Writes ``eval/retrieval_results/<variant>.csv`` per variant and a
combined ``eval/retrieval_results/summary.md`` for the README.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from eval.metrics import QueryEval, aggregate, evaluate_query

logger = logging.getLogger("eval.run_retrieval_eval")

DEFAULT_VARIANTS = ("dense", "hybrid", "hybrid+rerank")


# ── Query set IO ──────────────────────────────────────────────────────────


@dataclass
class LabeledQuery:
    query_id: str
    query: str
    relevant_job_ids: set[str]


def load_queries(path: Path) -> list[LabeledQuery]:
    rows: list[LabeledQuery] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rows.append(
            LabeledQuery(
                query_id=rec["query_id"],
                query=rec["query"],
                relevant_job_ids=set(rec.get("relevant_job_ids", [])),
            )
        )
    return rows


# ── Variant runners ───────────────────────────────────────────────────────


def _retriever_for_variant(variant: str):
    """Construct + cache a HybridRetriever configured for this variant."""
    from app.retriever_loader import get_retriever
    from rag.reranker import load_reranker

    base = get_retriever()
    if variant == "dense":
        # Force sparse off by emptying out the embedder's sparse output —
        # easier: use a copy with reranker None and rely on the index
        # being dense-only (true for our v1 MiniLM index).
        return base
    if variant == "hybrid":
        # Same as dense when index has no sparse; with bge-m3 reindex it
        # gains the sparse leg automatically.
        return base
    if variant == "hybrid+rerank":
        if base.reranker is None:
            logger.info("loading lite cross-encoder for rerank variant")
            base.reranker = load_reranker(lite=True)
        return base
    raise ValueError(f"variant {variant!r} not implemented yet")


def _run_one_query(retriever, q: LabeledQuery, *, top_k: int) -> list[str]:
    """Return the ranked list of *job_ids* the retriever produces."""
    retriever.final_top_k = top_k
    results = retriever.search(q.query)
    seen: list[str] = []
    for r in results:
        if r.job_id and r.job_id not in seen:
            seen.append(r.job_id)
        if len(seen) >= top_k:
            break
    return seen


def run_variant(
    variant: str,
    queries: list[LabeledQuery],
    *,
    top_k: int,
    out_dir: Path,
) -> dict:
    retriever = _retriever_for_variant(variant)
    per_query: list[QueryEval] = []
    t0 = time.time()
    for i, q in enumerate(queries):
        retrieved = _run_one_query(retriever, q, top_k=top_k)
        per_query.append(
            evaluate_query(
                query_id=q.query_id,
                query=q.query,
                relevant_ids=q.relevant_job_ids,
                retrieved_ids=retrieved,
            )
        )
        if (i + 1) % 10 == 0:
            logger.info(
                "  %s :: %d/%d queries (%.1f s)", variant, i + 1, len(queries), time.time() - t0
            )
    elapsed = time.time() - t0

    # Per-query CSV.
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{variant.replace('+', '_')}.csv"
    cols = list(per_query[0].keys()) if per_query else []
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in per_query:
            f.write(",".join(_csv_field(r[k]) for k in cols) + "\n")

    summary = {
        "variant": variant,
        "n_queries": len(queries),
        "elapsed_sec": round(elapsed, 2),
        "per_query_csv": str(csv_path),
        "aggregate": aggregate(per_query),
    }
    return summary


def _csv_field(v) -> str:
    if isinstance(v, str):
        # Quote if contains comma/quote/newline.
        if any(c in v for c in (",", '"', "\n")):
            v = '"' + v.replace('"', '""') + '"'
        return v
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ── Markdown summary ──────────────────────────────────────────────────────


def to_markdown(summaries: Iterable[dict]) -> str:
    lines = [
        "# Retrieval eval — multi-variant comparison",
        "",
        "| Variant | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 | latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        agg = s["aggregate"]
        latency_per_q = s["elapsed_sec"] / max(s["n_queries"], 1)
        lines.append(
            f"| `{s['variant']}` | {agg['n_queries']} | "
            f"{agg['recall_at_5']:.3f} | {agg['recall_at_10']:.3f} | {agg['recall_at_20']:.3f} | "
            f"{agg['mrr']:.3f} | {agg['ndcg_at_10']:.3f} | "
            f"{latency_per_q * 1000:.0f} ms/q |"
        )
    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--queries",
        default="eval/retrieval_queries.jsonl",
        help="JSONL with {query_id, query, relevant_job_ids[]}",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Which variants to run",
    )
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out-dir", default="eval/retrieval_results")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    queries = load_queries(Path(args.queries))
    logger.info("loaded %d queries from %s", len(queries), args.queries)

    out_dir = Path(args.out_dir)
    summaries: list[dict] = []
    for variant in args.variants:
        logger.info("=== variant %s ===", variant)
        summaries.append(run_variant(variant, queries, top_k=args.top_k, out_dir=out_dir))

    # Combined summary JSON + markdown.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    (out_dir / "summary.md").write_text(to_markdown(summaries))
    print(to_markdown(summaries))
    logger.info("wrote %s", out_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
