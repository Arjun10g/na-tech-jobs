"""Search tab — DuckDB-backed keyword + filter search over the curated parquet.

Phase 3 v0: substring match on title + company_name, with country and
role-family filters. Phase 5 swaps the substring match for hybrid (sparse +
dense) retrieval over the bge-m3 index.
"""

from __future__ import annotations

import logging
from typing import Any

import duckdb
import gradio as gr
import pandas as pd

from app.model_loader import get_curated_path

logger = logging.getLogger("app.tabs.search")

ROLE_FAMILIES = ["(any)", "DS", "DA", "DE", "MLE", "RS", "AS", "SWE-ML", "Manager", "Other"]
COUNTRIES = ["(any)", "US", "CA"]


def _query(
    keyword: str,
    country: str,
    role_family: str,
    only_disclosed: bool,
    limit: int,
) -> pd.DataFrame:
    parquet_path = get_curated_path()
    con = duckdb.connect()
    sql = f"""
        SELECT
            title,
            company_name,
            country,
            COALESCE(region, '') AS region,
            COALESCE(city, '') AS city,
            role_family_extracted AS role,
            seniority_extracted AS seniority,
            CASE WHEN salary_disclosed THEN
                printf('$%d - $%d', salary_min_usd_yearly, salary_max_usd_yearly)
            ELSE '—' END AS salary_usd_yearly,
            url
        FROM read_parquet('{parquet_path.as_posix()}')
        WHERE 1=1
    """
    params: dict[str, Any] = {}
    if keyword.strip():
        sql += (
            " AND (lower(title) LIKE lower($keyword) OR lower(company_name) LIKE lower($keyword))"
        )
        params["keyword"] = f"%{keyword.strip()}%"
    if country and country != "(any)":
        sql += " AND country = $country"
        params["country"] = country
    if role_family and role_family != "(any)":
        sql += " AND role_family_extracted = $role"
        params["role"] = role_family
    if only_disclosed:
        sql += " AND salary_disclosed = TRUE"
    sql += f" ORDER BY posted_at DESC NULLS LAST LIMIT {int(limit)}"
    return con.execute(sql, params).df()


def _search(
    keyword: str,
    country: str,
    role_family: str,
    only_disclosed: bool,
    limit: int,
) -> tuple[str, pd.DataFrame]:
    try:
        df = _query(keyword, country, role_family, only_disclosed, limit)
    except Exception as exc:  # noqa: BLE001
        logger.exception("search failed")
        return f"⚠️ search failed: {exc}", pd.DataFrame()
    if df.empty:
        return "_no matches_", pd.DataFrame()
    summary = f"**{len(df)}** results (most recent first)"
    return summary, df


def build_tab() -> gr.Tab:
    with gr.Tab("Search") as tab:
        gr.Markdown(
            "## Search the curated dataset\n\n"
            "Substring match on title + company name; filter by country and "
            "role family. Phase 5 swaps this for bge-m3 hybrid retrieval — "
            "this v0 is the placeholder + sanity check that the parquet is "
            "reachable."
        )

        with gr.Row():
            keyword = gr.Textbox(
                label="Keyword",
                placeholder='e.g. "machine learning"',
                lines=1,
                scale=3,
            )
            country = gr.Dropdown(label="Country", choices=COUNTRIES, value="(any)", scale=1)
            role_family = gr.Dropdown(
                label="Role family", choices=ROLE_FAMILIES, value="(any)", scale=1
            )
        with gr.Row():
            only_disclosed = gr.Checkbox(label="Disclosed salary only", value=False)
            limit = gr.Slider(label="Max results", minimum=10, maximum=100, step=10, value=25)
            search_btn = gr.Button("Search", variant="primary", scale=1)

        summary_md = gr.Markdown("_run a search…_")
        results = gr.Dataframe(
            headers=[
                "title",
                "company_name",
                "country",
                "region",
                "city",
                "role",
                "seniority",
                "salary_usd_yearly",
                "url",
            ],
            wrap=True,
            interactive=False,
            row_count=10,
        )

        search_btn.click(
            fn=_search,
            inputs=[keyword, country, role_family, only_disclosed, limit],
            outputs=[summary_md, results],
        )

    return tab
