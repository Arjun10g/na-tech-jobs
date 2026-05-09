"""Sample diverse rows from the curated parquet for LLM-propose labeling.

Per CLAUDE.md §7 the v1 metrics are weak-supervised (regex agreement). The
phase-4-followup plan: spawn Claude agents to **propose** labels, then a
human reviewer spot-checks via ``scripts.label_classifier --review``. The
output of the review is the eval set on which model cards report final F1.

This script just builds the *proposal input*: stratified samples saved as
JSONL shards under ``data/eval_proposals/<classifier>/``. Each row has::

    {"id", "title", "description_md", "regex_label"}

Run::

    uv run python -m scripts.build_eval_proposals seniority --n 250 --shard-size 50
    uv run python -m scripts.build_eval_proposals role_family --n 250 --shard-size 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("build_eval_proposals")

LABEL_COLUMNS: dict[str, str] = {
    "seniority": "seniority_extracted",
    "role_family": "role_family_extracted",
}
FALLBACK_LABELS: dict[str, str] = {
    "seniority": "mid",
    "role_family": "Other",
}
LABELS_BY_CLASSIFIER: dict[str, list[str]] = {
    "seniority": [
        "intern",
        "junior",
        "mid",
        "senior",
        "staff",
        "principal",
        "manager",
        "director",
        "exec",
    ],
    "role_family": [
        "DS",
        "DA",
        "DE",
        "MLE",
        "RS",
        "AS",
        "SWE-ML",
        "Manager",
        "Other",
    ],
}


def sample_for_review(
    df: pd.DataFrame,
    classifier: str,
    n: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Half on regex-fallback rows (the model's actual job — these have to
    generalize), half stratified across the explicit-match labels for
    class coverage.
    """
    label_col = LABEL_COLUMNS[classifier]
    fallback = FALLBACK_LABELS[classifier]

    n_fallback = n // 2
    n_explicit = n - n_fallback

    fallback_pool = df[df[label_col] == fallback]
    fallback_sample = fallback_pool.sample(
        n=min(n_fallback, len(fallback_pool)),
        random_state=seed,
    )

    explicit_chunks: list[pd.DataFrame] = []
    explicit_labels = [lbl for lbl in LABELS_BY_CLASSIFIER[classifier] if lbl != fallback]
    per_label = max(1, n_explicit // len(explicit_labels))
    for label in explicit_labels:
        sub = df[df[label_col] == label]
        if sub.empty:
            continue
        explicit_chunks.append(sub.sample(n=min(per_label, len(sub)), random_state=seed))
    explicit_sample = (
        pd.concat(explicit_chunks) if explicit_chunks else pd.DataFrame(columns=df.columns)
    )

    combined = pd.concat([fallback_sample, explicit_sample]).drop_duplicates(subset=["id"])
    return combined.sample(frac=1, random_state=seed).head(n)


def shard_to_jsonl(
    sample: pd.DataFrame,
    classifier: str,
    out_dir: Path,
    shard_size: int,
    description_chars: int = 1500,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    label_col = LABEL_COLUMNS[classifier]
    rows = sample.to_dict(orient="records")

    paths: list[Path] = []
    for i in range(0, len(rows), shard_size):
        shard = rows[i : i + shard_size]
        shard_path = out_dir / f"shard_{i // shard_size:02d}.jsonl"
        with shard_path.open("w") as f:
            for row in shard:
                desc = (row.get("description_md") or "")[:description_chars]
                payload = {
                    "id": row["id"],
                    "title": row.get("title"),
                    "company_name": row.get("company_name"),
                    "location_raw": row.get("location_raw"),
                    "description_md": desc,
                    "regex_label": row.get(label_col),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        paths.append(shard_path)
        logger.info("wrote shard :: %s (%d rows)", shard_path, len(shard))
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("classifier", choices=tuple(LABEL_COLUMNS))
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--shard-size", type=int, default=50)
    p.add_argument("--curated-path", default="data/curated/jobs.parquet")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to data/eval_proposals/<classifier>",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--description-chars", type=int, default=1500)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    classifier = args.classifier
    out_dir = Path(args.out_dir or f"data/eval_proposals/{classifier}")

    df = pd.read_parquet(args.curated_path)
    logger.info("loaded %d rows from %s", len(df), args.curated_path)

    sample = sample_for_review(df, classifier, args.n, seed=args.seed)
    logger.info("sampled %d rows for %s", len(sample), classifier)
    shard_to_jsonl(
        sample,
        classifier,
        out_dir,
        shard_size=args.shard_size,
        description_chars=args.description_chars,
    )
    logger.info("done :: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
