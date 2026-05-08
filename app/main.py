"""Gradio entrypoint for the na-tech-jobs Space.

Phase 3: salary prediction tab + curated-dataset search tab live. Matcher
(resume → top-k jobs) lands in Phase 5 alongside bge-m3 hybrid retrieval;
Analytics (NL→SQL) and Dashboard (drift) land in Phases 7-8.
"""

from __future__ import annotations

import gradio as gr

from app.tabs import salary, search

PROJECT_NAME = "na-tech-jobs"
TAGLINE = "A production ML platform for the North American senior tech-hiring market."
PHASE = "Phase 3 — first deployable build"


def status() -> str:
    return (
        f"**{PROJECT_NAME}** — {TAGLINE}\n\n"
        f"_{PHASE}._ Salary prediction (XGBoost on tabular features) and "
        "curated-dataset search are live below. The matcher + analytics + "
        "drift dashboard tabs land in later phases.\n\n"
        "**Links**\n"
        "- Source: https://github.com/Arjun10g/na-tech-jobs\n"
        "- Dataset: https://huggingface.co/datasets/arjun10g/na-tech-jobs\n"
        "- Model: https://huggingface.co/arjun10g/na-tech-jobs-salary-v1\n"
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title=PROJECT_NAME, theme=gr.themes.Soft()) as app:
        gr.Markdown(f"# {PROJECT_NAME}\n{TAGLINE}")
        with gr.Tab("Status"):
            gr.Markdown(status())
        salary.build_tab()
        search.build_tab()
        with gr.Tab("Matcher"):
            gr.Markdown("_Phase 5 — paste a resume, get ranked job matches._")
        with gr.Tab("Analytics"):
            gr.Markdown("_Phase 7 — NL→SQL over the curated dataset._")
        with gr.Tab("Dashboard"):
            gr.Markdown("_Phase 8 — drift, market trends, pipeline health._")
    return app


demo = build_app()

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
