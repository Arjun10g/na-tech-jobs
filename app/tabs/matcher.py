"""Matcher tab — natural-language query → ranked job matches + LLM rationale.

Per CLAUDE.md §8 the final stage of the retrieval pipeline summarizes the
top results with an LLM. This tab does:

  query → MiniLM-dense first-pass over Qdrant → optional rerank →
  parent-chunk hydration → top-K jobs → **LLM rationale** →
  results table + a 3-4 sentence "why these match" block.

The LLM call is best-effort: if the LLM isn't configured, the table
still renders, just without the rationale.

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


# ── LLM rationale ────────────────────────────────────────────────────────


_RATIONALE_SYSTEM = (
    "You are a candid career advisor helping a senior data scientist evaluate "
    "open job postings against their query. Be specific, concise, and honest. "
    "Cite jobs by company + a 1-3 word handle. Do not fabricate details."
)


def _build_rationale_prompt(query: str, rows: list[dict]) -> str:
    """Compact prompt — title, company, location, salary, role+seniority,
    plus a short snippet for the top 8 jobs."""
    lines = [f'User query: "{query.strip()}"\n', "Top matches retrieved:"]
    for i, r in enumerate(rows[:8], start=1):
        snippet = (r.get("snippet") or "").replace("`", "")
        if len(snippet) > 220:
            snippet = snippet[:217] + "…"
        lines.append(
            f"{i}. **{r['title']}** at {r['company']} — "
            f"{r['location']} ({r['country']}) — "
            f"{r['seniority']}/{r['role_family']}, "
            f"predicted {r['predicted_salary_usd_yr']}/yr. "
            f"_{snippet}_"
        )
    lines.append(
        "\nIn 3-4 sentences: which 1-2 jobs best match the query and why? "
        "Note any pattern across the set (salary range, geography, common "
        "skill mismatches). If a few jobs look like obvious mismatches, say so."
    )
    return "\n".join(lines)


def _llm_rationale(query: str, rows: list[dict]) -> str | None:
    """Generate the rationale via whichever LLM backend is configured.
    Returns None on any failure — caller renders without it."""
    try:
        from rag.nl2sql import default_llm

        llm = default_llm()
    except RuntimeError:
        return None
    try:
        prompt = _build_rationale_prompt(query, rows)
        return llm.generate(_RATIONALE_SYSTEM, prompt, max_tokens=400).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("matcher rationale failed :: %s", exc)
        return None


# ── Search handler ───────────────────────────────────────────────────────


def _run_query(
    query: str,
    country: str,
    role_family: str,
    seniority: str,
    min_salary: int | None,
    max_salary: int | None,
    top_k: int,
) -> tuple[pd.DataFrame, str, str]:
    """Returns (results_df, status_md, rationale_md)."""
    if not query or not query.strip():
        return pd.DataFrame(), "Type a query (or paste a resume blurb) to begin.", ""

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
        return pd.DataFrame(), msg, ""

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
        return pd.DataFrame(), f"Retrieval error: {exc}", ""

    if not results:
        return pd.DataFrame(), "No matches. Try broadening the filters.", ""

    rows: list[dict] = []
    for r in results:
        p = r.payload or {}
        snippet = (r.text or "").strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:277] + "…"
        url = p.get("url") or ""
        link_html = f'<a href="{url}" target="_blank" rel="noopener">apply ↗</a>' if url else "—"
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
                "link": link_html,
            }
        )
    df = pd.DataFrame(rows)
    summary = f"**{len(rows)} match(es)** for `{query.strip()[:120]}`."

    rationale = _llm_rationale(query, rows)
    rationale_md = f"### Why these jobs\n\n{rationale}" if rationale else ""

    return df, summary, rationale_md


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
            "## Matcher\n"
            "Type a natural-language query — or paste a resume blurb — and the "
            "hybrid-retrieval pipeline ranks the corpus by semantic match, then an "
            "LLM summarizes why these jobs fit. Filters apply at retrieval time."
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
        rationale_md = gr.Markdown("")
        results_df = gr.Dataframe(
            label="Top matches",
            headers=[
                "score",
                "title",
                "company",
                "location",
                "country",
                "seniority",
                "role_family",
                "predicted_salary_usd_yr",
                "skills",
                "snippet",
                "link",
            ],
            datatype=[
                "number",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "html",
            ],
            wrap=True,
            interactive=False,
        )
        run_btn.click(
            _run_query,
            inputs=[query, country, role_family, seniority, min_salary, max_salary, top_k],
            outputs=[results_df, summary_md, rationale_md],
        )
        gr.Examples(
            examples=EXAMPLES,
            inputs=[query, country, role_family, seniority, min_salary, max_salary, top_k],
            label="Examples",
        )
    return tab
