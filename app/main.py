"""Gradio entrypoint for the na-tech-jobs Space.

All five user-facing tabs live here: Salary, Search, Matcher,
Analytics, Dashboard. Build-time imports are kept minimal — each tab
defers heavy imports (torch, qdrant_client, sentence-transformers, etc.)
to first interaction so app startup stays fast.

UI is dark-themed by default — forced via Gradio's ?__theme=dark URL
arg at first paint. The accompanying CSS provides typography (system
font stack, tabular numerals) and tightens the dark-mode palette so
the contrast doesn't flatten readability.
"""

from __future__ import annotations

import gradio as gr

from app.tabs import analytics, dashboard, matcher, salary, search

PROJECT_NAME = "na-tech-jobs"
TAGLINE = "A production ML platform for the North American senior tech-hiring market."
PHASE = "Phase 8 — operational dashboard live"

# Force dark mode on first paint. Gradio respects the `?__theme=dark`
# query param + class on <body>; this snippet sets both whenever the
# user lands on the Space. Saves a manual toggle.
_FORCE_DARK_JS = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.replace(url.toString());
    }
}
"""

# Typography + dark-mode color tweaks. All overrides scoped to
# .gradio-container so they don't leak into Gradio chrome.
_CSS = """
:root {
    --na-bg: #0b0d12;
    --na-bg-elevated: #11141b;
    --na-border: #232735;
    --na-text: #e7e9ee;
    --na-text-muted: #9aa0ac;
    --na-accent: #818cf8; /* indigo-400, gentler than indigo-500 in dark */
}

.gradio-container,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .markdown,
.gradio-container .markdown p {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.gradio-container { max-width: 1280px !important; }

/* Headings */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    font-weight: 600;
    letter-spacing: -0.015em;
    line-height: 1.25;
    color: var(--na-text);
}
.gradio-container h1 { font-size: 2.0rem; margin-bottom: 0.25rem; }
.gradio-container h2 { font-size: 1.4rem; margin-top: 1.4rem; margin-bottom: 0.5rem; }
.gradio-container h3 { font-size: 1.1rem; margin-top: 1.0rem; margin-bottom: 0.4rem; }
.gradio-container p { line-height: 1.55; }

/* Mono for code + tabular numerals for tables */
.gradio-container code,
.gradio-container pre,
.gradio-container .code-wrap pre {
    font-family: "SFMono-Regular", "Menlo", "Consolas", "Liberation Mono", monospace !important;
    font-size: 0.92em;
}
.gradio-container table { font-feature-settings: "tnum" 1, "lnum" 1; }

/* Dark-mode overrides — only kick in when Gradio has applied .dark */
.dark .gradio-container,
.gradio-container.dark {
    background: var(--na-bg) !important;
}
.dark .gradio-container .block,
.dark .gradio-container .form,
.dark .gradio-container .panel {
    background: var(--na-bg-elevated) !important;
    border-color: var(--na-border) !important;
}
.dark .gradio-container input,
.dark .gradio-container textarea,
.dark .gradio-container select {
    background: var(--na-bg) !important;
    color: var(--na-text) !important;
    border-color: var(--na-border) !important;
}
.dark .gradio-container .tab-nav button {
    color: var(--na-text-muted) !important;
}
.dark .gradio-container .tab-nav button.selected {
    color: var(--na-accent) !important;
    border-bottom-color: var(--na-accent) !important;
}
.dark .gradio-container .markdown,
.dark .gradio-container .markdown p,
.dark .gradio-container .markdown li {
    color: var(--na-text) !important;
}
.dark .gradio-container .markdown a { color: var(--na-accent) !important; }

/* Project header band */
#na-tech-jobs-header h1 { margin-bottom: 0.1em; }
#na-tech-jobs-header p { color: var(--na-text-muted); margin-top: 0; }

/* Tighten dataframe rows. Per-tab `wrap=` controls cell wrapping;
   we just tighten padding/typography globally and ensure the table
   container scrolls horizontally for wide result sets. */
.gradio-container .table-wrap td,
.gradio-container .table-wrap th,
.gradio-container .svelte-virtual-table-viewport td,
.gradio-container .svelte-virtual-table-viewport th {
    padding: 0.35rem 0.55rem !important;
    font-size: 0.9em;
    line-height: 1.35;
    vertical-align: top;
}
/* Stop a single long-text cell from stretching its column to the full
   container width; enforce a ceiling and let overflow scroll. */
.gradio-container .table-wrap td { max-width: 420px; }
.gradio-container .table-wrap { overflow-x: auto !important; }
/* Dark-mode dataframe — force light text on dark cells. Gradio's
   default dataframe in dark mode is low-contrast; this fixes it. */
.dark .gradio-container .table-wrap,
.dark .gradio-container .table-wrap table,
.dark .gradio-container .table-wrap td,
.dark .gradio-container .table-wrap th,
.dark .gradio-container .svelte-virtual-table-viewport,
.dark .gradio-container .svelte-virtual-table-viewport td,
.dark .gradio-container .svelte-virtual-table-viewport th {
    color: var(--na-text) !important;
    background: var(--na-bg-elevated) !important;
}
.dark .gradio-container .table-wrap th,
.dark .gradio-container .svelte-virtual-table-viewport th {
    background: var(--na-bg) !important;
    color: var(--na-text-muted) !important;
    font-weight: 600;
    border-bottom: 1px solid var(--na-border) !important;
}
.dark .gradio-container .table-wrap tr:hover td {
    background: #1a1f2b !important;
}
/* Links inside dataframe cells (matcher URL column) */
.gradio-container .table-wrap td a,
.gradio-container .svelte-virtual-table-viewport td a {
    color: var(--na-accent) !important;
    text-decoration: underline;
}
.gradio-container .table-wrap td a:hover {
    text-decoration: none;
}

/* Primary button gets the accent color in dark */
.dark .gradio-container button.primary,
.dark .gradio-container button[variant="primary"] {
    background: var(--na-accent) !important;
    color: #0b0d12 !important;
    border: 0 !important;
    font-weight: 600;
}
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
        neutral_hue="zinc",
    )
    with gr.Blocks(
        title=PROJECT_NAME,
        theme=theme,
        css=_CSS,
        js=_FORCE_DARK_JS,
    ) as app:
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
