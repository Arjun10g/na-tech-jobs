"""Matcher tab — natural-language query → ranked job matches.

Phase 5 v0: text query (or pasted resume). The flow goes
query → MiniLM-dense first-pass over Qdrant → optional rerank →
parent-chunk hydration → top-K jobs displayed with predicted salary +
seniority + role family + a contributing snippet.

PDF resume parsing lands in v1.1 once the matcher is validated end-to-end.
"""

from __future__ import annotations

import logging

import gradio as gr
import pandas as pd

logger = logging.getLogger("app.tabs.matcher")

ROLE_FAMILIES = ["(any)", "AS", "DA", "DE", "DS", "MLE", "RS", "SWE-ML"]
SENIORITY_LEVELS = ["(any)", "junior", "senior", "staff", "principal", "manager", "director"]
COUNTRIES = ["(any)", "US", "CA"]


def _format_money(v) -> str:
    if v is None or v != v:  # noqa: PLR0124 — NaN check
        return "—"
    try:
        return f"${int(round(float(v))):,}"
    except (TypeError, ValueError):
        return "—"


def _run_query(
    query: str,
    country: str,
    role_family: str,
    seniority: str,
    min_salary: int | None,
    max_salary: int | None,
    top_k: int,
) -> tuple[pd.DataFrame, str]:
    if not query or not query.strip():
        return pd.DataFrame(), "Type a query (or paste a resume blurb) to begin."

    try:
        from app.retriever_loader import get_retriever
        from rag.pipeline import build_filter

        retriever = get_retriever()
    except (FileNotFoundError, RuntimeError) as exc:
        msg = (
            f"Retriever not ready: {exc}\n\n"
            "On a fresh checkout run:\n"
            "    uv run python -m scripts.index_jobs --lite"
        )
        return pd.DataFrame(), msg

    qfilter = build_filter(
        countries=[country] if country != "(any)" else None,
        seniority_labels=[seniority] if seniority != "(any)" else None,
        role_families=[role_family] if role_family != "(any)" else None,
        min_predicted_salary_usd=min_salary if min_salary else None,
        max_predicted_salary_usd=max_salary if max_salary else None,
    )

    try:
        retriever.final_top_k = max(1, min(top_k or 10, 25))
        results = retriever.search(query.strip(), qdrant_filter=qfilter)
    except Exception as exc:  # noqa: BLE001
        logger.exception("retrieval failed")
        return pd.DataFrame(), f"Retrieval error: {exc}"

    if not results:
        return pd.DataFrame(), "No matches. Try broadening the filters."

    rows: list[dict] = []
    for r in results:
        p = r.payload or {}
        snippet = (r.text or "").strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:277] + "…"
        rows.append(
            {
                "score": round(r.score, 3),
                "title": p.get("title") or "(no title)",
                "company": p.get("company_name") or "—",
                "location": (p.get("city") or p.get("region") or p.get("country") or "—"),
                "country": p.get("country") or "—",
                "seniority": p.get("seniority_label_v1") or p.get("seniority_extracted") or "—",
                "role_family": p.get("role_family_v1") or p.get("role_family_extracted") or "—",
                "predicted_salary_usd_yr": _format_money(p.get("predicted_salary_usd_v1")),
                "skills": ", ".join((p.get("extracted_skills_v1") or [])[:6]),
                "snippet": snippet,
                "url": p.get("url") or "",
            }
        )
    df = pd.DataFrame(rows)
    summary = f"**{len(rows)} match(es)** for `{query.strip()[:120]}`."
    return df, summary


# ── Gradio tab ────────────────────────────────────────────────────────────


EXAMPLES = [
    [
        "Senior MLE building production recommender systems with PyTorch, Spark, GCP.",
        "(any)",
        "MLE",
        "senior",
        None,
        None,
        10,
    ],
    [
        "Data scientist with strong SQL + experimentation, dashboards, A/B testing in fintech.",
        "US",
        "DS",
        "(any)",
        None,
        None,
        10,
    ],
    [
        "Staff data engineer, dbt + Snowflake + Airflow, lakehouse architecture.",
        "(any)",
        "DE",
        "staff",
        None,
        None,
        10,
    ],
    ["Toronto-based ML researcher, LLM alignment, NLP.", "CA", "RS", "(any)", None, None, 10],
]


def build_tab() -> gr.Tab:
    with gr.Tab("Matcher") as tab:
        gr.Markdown(
            "## Matcher\n\n"
            "Hybrid retrieval over the indexed job corpus. Type a natural-language "
            "query (or paste a resume blurb) and apply filters as needed.\n\n"
            "_Default backend: MiniLM dense. Cross-encoder rerank is OFF by default "
            "to keep latency under 1 s; enable via `RAG_RERANKER=lite` env var._"
        )
        with gr.Row():
            with gr.Column(scale=2):
                query = gr.Textbox(
                    label="Query / Resume blurb",
                    lines=4,
                    placeholder="e.g. 'Senior MLE building recommender systems on GCP'",
                )
            with gr.Column(scale=1):
                country = gr.Dropdown(COUNTRIES, value="(any)", label="Country")
                role_family = gr.Dropdown(ROLE_FAMILIES, value="(any)", label="Role family")
                seniority = gr.Dropdown(SENIORITY_LEVELS, value="(any)", label="Seniority")
        with gr.Row():
            min_salary = gr.Number(label="Min predicted salary USD/yr", precision=0, value=None)
            max_salary = gr.Number(label="Max predicted salary USD/yr", precision=0, value=None)
            top_k = gr.Slider(1, 25, value=10, step=1, label="Top-K results")
        run_btn = gr.Button("Match", variant="primary")
        summary_md = gr.Markdown("")
        results_df = gr.Dataframe(
            label="Top matches",
            wrap=True,
            interactive=False,
        )
        run_btn.click(
            _run_query,
            inputs=[query, country, role_family, seniority, min_salary, max_salary, top_k],
            outputs=[results_df, summary_md],
        )
        gr.Examples(
            examples=EXAMPLES,
            inputs=[query, country, role_family, seniority, min_salary, max_salary, top_k],
            label="Examples",
        )
    return tab
