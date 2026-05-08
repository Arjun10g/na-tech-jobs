"""Gradio entrypoint for the na-tech-jobs Space.

Phase 0: hello-world shell. Tabs (matcher, search, analytics, dashboard) get
filled in across phases 3-8.
"""

from __future__ import annotations

import os

import gradio as gr

PROJECT_NAME = "na-tech-jobs"
TAGLINE = "A production ML platform for the North American senior tech-hiring market."
PHASE = "Phase 0 — scaffolding"


def status() -> str:
    return (
        f"**{PROJECT_NAME}** is live.\n\n"
        f"_{TAGLINE}_\n\n"
        f"Current build: **{PHASE}**.\n\n"
        f"The matcher, search, analytics, and dashboard tabs land in later phases."
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title=PROJECT_NAME, theme=gr.themes.Soft()) as app:
        gr.Markdown(f"# {PROJECT_NAME}\n{TAGLINE}")
        with gr.Tab("Status"):
            gr.Markdown(status())
        with gr.Tab("Matcher"):
            gr.Markdown("_Phase 5 — paste a resume, get ranked job matches._")
        with gr.Tab("Search"):
            gr.Markdown("_Phase 5/7 — natural-language search over indexed postings._")
        with gr.Tab("Analytics"):
            gr.Markdown("_Phase 7 — NL→SQL over the curated dataset._")
        with gr.Tab("Dashboard"):
            gr.Markdown("_Phase 8 — drift, market trends, pipeline health._")
    return app


if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", 7860))
    build_app().launch(server_name="0.0.0.0", server_port=server_port, ssr_mode=False)
