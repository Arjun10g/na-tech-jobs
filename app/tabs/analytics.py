"""Analytics tab — natural-language → DuckDB SQL → results, with the
mandatory CLAUDE.md §11 safety layer (sqlglot allowlist + row/time caps).

LLM backend is selected automatically: Anthropic Claude when
``ANTHROPIC_API_KEY`` is set, else HF Inference for Qwen2.5-7B when
``HF_TOKEN`` is set, else the tab shows a "configure your LLM" message
and the schema panel still renders.

Per CLAUDE.md §8 the executed SQL is **always** shown to the user
alongside the result so they can verify what the LLM actually ran.
"""

from __future__ import annotations

import logging

import gradio as gr
import pandas as pd

logger = logging.getLogger("app.tabs.analytics")


def _resolve_curated_path():
    """Prefer the Phase 4 enriched parquet (versioned predictions) when
    present; fall back to the bare curated parquet."""
    from pathlib import Path

    enriched = Path("data/curated_enriched/jobs.parquet")
    if enriched.exists():
        return enriched
    return Path("data/curated/jobs.parquet")


def _run_nl2sql(question: str) -> tuple[str, str, pd.DataFrame]:
    """Returns (status_md, sql, results_df)."""
    if not question or not question.strip():
        return ("Type a question to begin.", "", pd.DataFrame())

    from rag.nl2sql import nl_to_sql, serialize_result  # noqa: F401

    curated_path = _resolve_curated_path()
    if not curated_path.exists():
        return (
            f"❌ Curated parquet missing at `{curated_path}`. Run "
            "`uv run python -m curated.enrich` first.",
            "",
            pd.DataFrame(),
        )

    try:
        result = nl_to_sql(question.strip(), curated_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("nl_to_sql crashed")
        return (f"❌ Pipeline crashed: {exc}", "", pd.DataFrame())

    if result.error:
        sql_block = result.sql or ""
        return (
            f"⚠️  {result.error}",
            sql_block,
            pd.DataFrame(),
        )

    return (
        f"**{result.n_rows} row(s)** for `{question.strip()[:120]}`.",
        result.sql or "",
        result.rows if result.rows is not None else pd.DataFrame(),
    )


# ── Gradio tab ────────────────────────────────────────────────────────────


EXAMPLES: list[list[str]] = [
    ["What's the median predicted salary for senior MLE roles in the US?"],
    ["How many DE jobs are open in Toronto right now?"],
    ["Top 10 companies by number of senior data scientist postings."],
    ["Distribution of role_family_v1 across countries."],
    ["What's the average disclosed salary range for staff-level roles?"],
    ["Jobs posted in the last 30 days with predicted salary above $250k."],
]


def build_tab() -> gr.Tab:
    from rag.nl2sql import schema_description

    with gr.Tab("Analytics") as tab:
        gr.Markdown(
            "## Analytics — Natural-language SQL\n\n"
            "Ask a question in plain English. An LLM writes DuckDB SQL "
            "against the curated jobs table; a sqlglot-based safety "
            "layer (CLAUDE.md §8 + §11) rejects anything that isn't a "
            "read-only SELECT over the allowed columns. The executed "
            "SQL is always shown so you can verify it.\n\n"
            "_Backend: Anthropic Claude (preferred when `ANTHROPIC_API_KEY` "
            "is set), else HF Inference / Qwen2.5-7B (when `HF_TOKEN` is "
            "set). Results capped at 1000 rows / 5 s._"
        )
        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="Question",
                    lines=2,
                    placeholder="e.g. What's the median predicted salary for senior MLEs in the US?",
                )
                run_btn = gr.Button("Ask", variant="primary")
                status_md = gr.Markdown("")
                executed_sql = gr.Code(label="Executed SQL", language="sql")
                results_df = gr.Dataframe(label="Results", interactive=False, wrap=True)
            with gr.Column(scale=1):
                gr.Markdown("### Schema (read-only)")
                gr.Markdown("```\n" + schema_description() + "\n```")

        run_btn.click(
            _run_nl2sql,
            inputs=[question],
            outputs=[status_md, executed_sql, results_df],
        )
        gr.Examples(examples=EXAMPLES, inputs=[question], label="Examples")
    return tab
