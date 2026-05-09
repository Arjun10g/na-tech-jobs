"""Build a labeled retrieval-eval query set.

Two query types in the output:

1. **Title-as-query**: sample N jobs, use the title as the query, and
   take *all* curated jobs with the same (normalized title, role_family,
   country) as the gold set. This rewards retrieval that finds
   functionally-equivalent jobs at other companies, not just the exact
   sampled doc.
2. **Role+seniority queries**: hand-crafted natural-language patterns
   ("Senior MLE in Toronto", "Staff data engineer remote in US") with
   gold pools defined by classifier-derived
   ``role_family_v1``/``seniority_label_v1`` filters on the enriched
   parquet.

Output: ``eval/retrieval_queries.jsonl``. Each line:

    {"query_id": "...", "query": "...", "relevant_job_ids": [...],
     "kind": "title-self" | "role-seniority", "stratum": "..."}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger("build_retrieval_queries")


def _normalize_title(t: str) -> str:
    """Normalize a title for matching: lowercase, strip parens/brackets,
    drop trailing seniority/level qualifiers, collapse whitespace.
    Keeps enough detail to distinguish "Senior MLE" from "Senior DE"."""
    t = (t or "").lower()
    t = re.sub(r"[\(\[].*?[\)\]]", "", t)  # parenthetical content
    t = re.sub(r"\b(?:i{1,3}|iv|v|vi)\b", "", t)  # roman numeral level
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _build_title_queries(
    df: pd.DataFrame,
    n: int,
    *,
    seed: int = 42,
    min_pool: int = 2,
    max_pool: int = 50,
) -> list[dict]:
    """Sample N jobs, compute their gold-pool by (norm_title, country)."""
    df = df.assign(_norm_title=df["title"].fillna("").map(_normalize_title))
    counts = df["_norm_title"].value_counts()

    # Pool only matters if there are at least `min_pool` jobs with the same
    # normalized title — otherwise the only gold doc is the one itself,
    # which makes the metric trivial.
    valid_titles = counts[counts >= min_pool].index.tolist()
    df_valid = df[df["_norm_title"].isin(valid_titles)]
    if len(df_valid) < n:
        n = len(df_valid)
    sampled = df_valid.sample(n=n, random_state=seed)

    out: list[dict] = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        norm = row["_norm_title"]
        pool = df[(df["_norm_title"] == norm) & (df["country"] == row["country"])]
        relevant = pool["id"].tolist()[:max_pool]
        out.append(
            {
                "query_id": f"title-{i:03d}",
                "query": row["title"],
                "relevant_job_ids": relevant,
                "kind": "title-self",
                "stratum": f"title={norm[:40]} country={row['country']}",
            }
        )
    return out


# Hand-crafted natural-language query templates. The gold pool for each
# is computed by classifier filters on the enriched parquet — ``role`` /
# ``seniority`` / ``country`` constraints. ``min_pool`` ensures we don't
# emit queries with empty gold sets.
ROLE_SENIORITY_TEMPLATES: list[dict] = [
    {
        "q": "Senior machine learning engineer building production recommender systems",
        "role": "MLE",
        "seniority": "senior",
    },
    {
        "q": "Staff ML engineer working on large-scale model training infrastructure",
        "role": "MLE",
        "seniority": "staff",
    },
    {
        "q": "Principal data scientist leading experimentation and causal inference work",
        "role": "DS",
        "seniority": "principal",
    },
    {
        "q": "Senior data scientist with strong SQL and A/B testing experience",
        "role": "DS",
        "seniority": "senior",
    },
    {
        "q": "Staff data engineer designing modern data lakehouse with dbt and Snowflake",
        "role": "DE",
        "seniority": "staff",
    },
    {
        "q": "Senior data engineer building Spark / Airflow pipelines at scale",
        "role": "DE",
        "seniority": "senior",
    },
    {
        "q": "Research scientist focused on LLM alignment and AI safety",
        "role": "RS",
        "seniority": "senior",
    },
    {
        "q": "Applied scientist bridging ML research and product applications",
        "role": "AS",
        "seniority": "senior",
    },
    {
        "q": "Staff software engineer on ML systems and inference infrastructure",
        "role": "SWE-ML",
        "seniority": "staff",
    },
    {
        "q": "Director of data science leading cross-functional analytics teams",
        "role": "DS",
        "seniority": "director",
    },
    {
        "q": "Senior MLE in Toronto building NLP products",
        "role": "MLE",
        "seniority": "senior",
        "country": "CA",
    },
    {
        "q": "Staff data engineer remote-friendly in Canada with cloud expertise",
        "role": "DE",
        "seniority": "staff",
        "country": "CA",
    },
    {"q": "Senior research engineer at a frontier AI lab", "role": "RS", "seniority": "senior"},
    {
        "q": "Applied scientist for personalization and ranking systems",
        "role": "AS",
        "seniority": "senior",
    },
    {
        "q": "Senior data analyst with dashboarding and BI experience",
        "role": "DA",
        "seniority": "senior",
    },
    {
        "q": "Junior data engineer eager to learn modern data stack",
        "role": "DE",
        "seniority": "junior",
    },
    {
        "q": "Manager of machine learning engineering, leading team of MLEs",
        "role": "MLE",
        "seniority": "manager",
    },
    {
        "q": "Staff machine learning engineer for generative AI / LLM applications",
        "role": "MLE",
        "seniority": "staff",
    },
    {
        "q": "Senior data scientist in fintech building risk and fraud models",
        "role": "DS",
        "seniority": "senior",
    },
    {
        "q": "Senior software engineer working on training infrastructure for LLMs",
        "role": "SWE-ML",
        "seniority": "senior",
    },
]


def _build_role_seniority_queries(
    df: pd.DataFrame,
    *,
    min_pool: int = 5,
    max_pool: int = 100,
    has_classifier_columns: bool,
) -> list[dict]:
    out: list[dict] = []
    role_col = "role_family_v1" if has_classifier_columns else "role_family_extracted"
    sen_col = "seniority_label_v1" if has_classifier_columns else "seniority_extracted"

    for i, t in enumerate(ROLE_SENIORITY_TEMPLATES):
        mask = (df[role_col] == t["role"]) & (df[sen_col] == t["seniority"])
        if "country" in t:
            mask &= df["country"] == t["country"]
        pool = df.loc[mask]
        if len(pool) < min_pool:
            logger.warning(
                "skipping query (pool too small): %r → %d matches",
                t["q"],
                len(pool),
            )
            continue
        relevant = pool["id"].tolist()[:max_pool]
        stratum = f"role={t['role']} sen={t['seniority']}" + (
            f" country={t['country']}" if "country" in t else ""
        )
        out.append(
            {
                "query_id": f"role-{i:03d}",
                "query": t["q"],
                "relevant_job_ids": relevant,
                "kind": "role-seniority",
                "stratum": stratum,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--curated-path",
        default=None,
        help="Defaults to data/curated_enriched/jobs.parquet "
        "(falls back to data/curated/jobs.parquet)",
    )
    p.add_argument("--n-title-queries", type=int, default=30)
    p.add_argument("--out-path", default="eval/retrieval_queries.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    curated_path = Path(
        args.curated_path
        or (
            "data/curated_enriched/jobs.parquet"
            if Path("data/curated_enriched/jobs.parquet").exists()
            else "data/curated/jobs.parquet"
        )
    )
    df = pd.read_parquet(curated_path)
    logger.info("loaded %d jobs from %s", len(df), curated_path)

    has_classifier = "role_family_v1" in df.columns and "seniority_label_v1" in df.columns

    title_qs = _build_title_queries(df, args.n_title_queries, seed=args.seed)
    role_qs = _build_role_seniority_queries(df, has_classifier_columns=has_classifier)
    queries = title_qs + role_qs

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    logger.info(
        "wrote %d queries → %s (title=%d, role=%d)",
        len(queries),
        out_path,
        len(title_qs),
        len(role_qs),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
