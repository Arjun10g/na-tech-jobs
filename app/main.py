"""Gradio entrypoint for the na-tech-jobs Space.

Phase 5: salary prediction + curated keyword search + **hybrid RAG matcher**
live. Analytics (NL→SQL) and Dashboard (drift) land in Phases 7-8.
"""

from __future__ import annotations

import gradio as gr

from app.tabs import analytics, matcher, salary, search

PROJECT_NAME = "na-tech-jobs"
TAGLINE = "A production ML platform for the North American senior tech-hiring market."
PHASE = "Phase 7 — NL→SQL analytics live (matcher + analytics + salary + search)"


def status() -> str:
    return (
        f"**{PROJECT_NAME}** — {TAGLINE}\n\n"
        f"_{PHASE}._ Salary prediction, curated keyword search, and the "
        "hybrid-retrieval matcher are all live. NL→SQL analytics and the "
        "drift dashboard land in Phases 7-8.\n\n"
        "**Links**\n"
        "- Source: https://github.com/Arjun10g/na-tech-jobs\n"
        "- Dataset: https://huggingface.co/datasets/arjun10g/na-tech-jobs\n"
        "- Models:\n"
        "  - https://huggingface.co/arjun10g/na-tech-jobs-salary-v1\n"
        "  - https://huggingface.co/arjun10g/na-tech-jobs-seniority-v1\n"
        "  - https://huggingface.co/arjun10g/na-tech-jobs-role_family-v1\n"
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title=PROJECT_NAME, theme=gr.themes.Soft()) as app:
        gr.Markdown(f"# {PROJECT_NAME}\n{TAGLINE}")
        with gr.Tab("Status"):
            gr.Markdown(status())
        salary.build_tab()
        search.build_tab()
        matcher.build_tab()
        analytics.build_tab()
        with gr.Tab("Dashboard"):
            gr.Markdown("_Phase 8 — drift, market trends, pipeline health._")
    return app


demo = build_app()

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
