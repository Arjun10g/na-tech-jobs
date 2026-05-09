"""Gradio entrypoint for the na-tech-jobs Space.

All five user-facing tabs live here: Salary, Search, Matcher,
Analytics, Dashboard. Build-time imports are kept minimal — each tab
defers heavy imports (torch, qdrant_client, sentence-transformers, etc.)
to first interaction so app startup stays fast.
"""

from __future__ import annotations

import gradio as gr

from app.tabs import analytics, dashboard, matcher, salary, search

PROJECT_NAME = "na-tech-jobs"
TAGLINE = "A production ML platform for the North American senior tech-hiring market."
PHASE = "Phase 8 — operational dashboard live"


# Cohesive typographic baseline. System font stack picks SF on macOS,
# Segoe on Windows, Roboto on Android, sans-serif everywhere else.
# Tabular numerals make all the numeric-heavy tables in Dashboard +
# Matcher line up cleanly.
_CSS = """
.gradio-container,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container .markdown,
.gradio-container .markdown p {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.gradio-container { max-width: 1280px !important; }
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    font-weight: 600;
    letter-spacing: -0.015em;
    line-height: 1.25;
}
.gradio-container h1 { font-size: 2.0rem; margin-bottom: 0.25rem; }
.gradio-container h2 { font-size: 1.4rem; margin-top: 1.4rem; margin-bottom: 0.5rem; }
.gradio-container h3 { font-size: 1.1rem; margin-top: 1.0rem; margin-bottom: 0.4rem; }
.gradio-container p { line-height: 1.55; }
.gradio-container code,
.gradio-container pre {
    font-family: "SFMono-Regular", "Menlo", "Consolas", "Liberation Mono", monospace !important;
    font-size: 0.92em;
}
.gradio-container table { font-feature-settings: "tnum" 1, "lnum" 1; }
.gradio-container button.lg { font-weight: 600; }

/* Project header band */
#na-tech-jobs-header h1 { margin-bottom: 0.1em; }
#na-tech-jobs-header p { color: var(--body-text-color-subdued); margin-top: 0; }
"""


def status() -> str:
    return (
        f"### What this is\n"
        f"_{PHASE}._  All five tabs are live — Salary, Search, Matcher, "
        f"Analytics, Dashboard.\n\n"
        "### Useful links\n"
        "- **Source**: https://github.com/Arjun10g/na-tech-jobs\n"
        "- **Dataset**: https://huggingface.co/datasets/arjun10g/na-tech-jobs\n"
        "- **Models**:\n"
        "  - https://huggingface.co/arjun10g/na-tech-jobs-salary-v1\n"
        "  - https://huggingface.co/arjun10g/na-tech-jobs-seniority-v1\n"
        "  - https://huggingface.co/arjun10g/na-tech-jobs-role_family-v1\n"
        "  - https://huggingface.co/arjun10g/na-tech-jobs-skills-v1\n"
    )


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
    )
    with gr.Blocks(title=PROJECT_NAME, theme=theme, css=_CSS) as app:
        with gr.Group(elem_id="na-tech-jobs-header"):
            gr.Markdown(f"# {PROJECT_NAME}\n{TAGLINE}")
        with gr.Tab("Status"):
            gr.Markdown(status())
        salary.build_tab()
        search.build_tab()
        matcher.build_tab()
        analytics.build_tab()
        dashboard.build_tab()
    return app


demo = build_app()

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
