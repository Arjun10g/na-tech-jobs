"""Dashboard tab — drift + pipeline health + market trends.

Per CLAUDE.md §10 Phase 8 this is the single-page operational view:

- **Pipeline health card**: last ingest / curated build / enrichment
  timestamps + per-extractor counts. Falls back to deriving from the
  curated parquet when no ingestion_stats.json is present yet.
- **Headline numbers**: jobs-on-corpus, companies, country split,
  disclosure rate, median predicted vs. disclosed salary.
- **Market-trend tables**: salary distribution by role × seniority, top
  companies by posting count, role-family share by country, top skills.
- **Drift report**: HTML iframe of the latest weekly Evidently report
  if one exists, plus the slim metrics card. Empty-state when no drift
  report has run yet.

Everything is computed on demand — no caching.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import gradio as gr
import pandas as pd

logger = logging.getLogger("app.tabs.dashboard")

DRIFT_REPORTS_DIR = Path("reports/drift")


def _format_money(v: float | int | None) -> str:
    if v is None:
        return "—"
    try:
        return f"${int(round(float(v))):,}"
    except (TypeError, ValueError):
        return "—"


def _headline_card() -> str:
    from monitoring.market_trends import headline_numbers

    try:
        h = headline_numbers()
    except Exception as exc:  # noqa: BLE001
        return f"_Could not compute headline numbers: {exc}_"
    return (
        f"### Corpus snapshot\n\n"
        f"- **Active jobs**: {h['n_jobs_active']:,}\n"
        f"- **Companies**: {h['n_companies']:,}\n"
        f"- **US / CA split**: {h['n_us']:,} / {h['n_ca']:,}\n"
        f"- **Salary-disclosed jobs**: {h['n_salary_disclosed']:,} "
        f"({h['salary_disclosure_rate'] * 100:.1f}%)\n"
        f"- **Median disclosed salary**: {_format_money(h['median_disclosed_salary_usd'])}/yr\n"
        f"- **Median predicted salary** (model v1, all rows): "
        f"{_format_money(h['median_predicted_salary_usd'])}/yr\n"
    )


def _pipeline_health_card() -> str:
    try:
        from monitoring.pipeline_health import collect_health, to_summary_md

        return to_summary_md(collect_health())
    except Exception as exc:  # noqa: BLE001
        return f"_Pipeline health unavailable: {exc}_"


def _salary_distribution_df() -> pd.DataFrame:
    from monitoring.market_trends import salary_distribution

    return salary_distribution().round({"p25": 0, "median": 0, "p75": 0})


def _top_companies_df() -> pd.DataFrame:
    from monitoring.market_trends import top_companies

    return top_companies(limit=20)


def _role_family_share_df() -> pd.DataFrame:
    from monitoring.market_trends import role_family_share

    return role_family_share()


def _top_skills_df() -> pd.DataFrame:
    from monitoring.market_trends import top_skills

    return top_skills(limit=30)


# ── Drift ────────────────────────────────────────────────────────────────


def _latest_drift() -> tuple[str | None, dict | None]:
    """Return (html_path, metrics_dict) for the most recent drift report."""
    if not DRIFT_REPORTS_DIR.exists():
        return None, None
    htmls = sorted(DRIFT_REPORTS_DIR.glob("*.html"))
    if not htmls:
        return None, None
    latest_html = htmls[-1]
    metrics_path = latest_html.with_suffix(".metrics.json")
    metrics = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = None
    return str(latest_html), metrics


def _drift_summary_md(metrics: dict | None) -> str:
    if not metrics:
        return (
            "_No drift report yet. Run "
            "`uv run python -m monitoring.drift --synthetic-split` for "
            "a v1 demo, or wait for two real snapshots to land._"
        )
    breach = metrics.get("priority_breach")
    icon = "🚨" if breach else "✅"
    parts = [
        f"### Drift report — `{metrics.get('snapshot_date', 'unknown')}` {icon}",
        f"- **Drift share**: {metrics.get('drift_share', 0.0):.2%} "
        f"({metrics.get('columns_drifted_count', 0)} of "
        f"{metrics.get('columns_total', 0)} tracked features drifted)",
        f"- **Reference rows**: {metrics.get('n_reference', 0):,} | "
        f"**Current rows**: {metrics.get('n_current', 0):,}",
        f"- **Priority retrain**: "
        f"{'YES — flagged for next monthly retrain' if breach else 'no breach'}",
    ]
    if metrics.get("breached_features"):
        parts.append(
            "- **Breached features**: " + ", ".join(f"`{c}`" for c in metrics["breached_features"])
        )
    drifted = metrics.get("columns_drifted") or []
    if drifted:
        parts.append("- **All drifted columns**: " + ", ".join(f"`{c}`" for c in drifted))
    return "\n".join(parts)


def _drift_html(html_path: str | None) -> str:
    if not html_path or not Path(html_path).exists():
        return ""
    # Embed inline rather than via iframe so Gradio's sandbox doesn't
    # block it.
    try:
        return Path(html_path).read_text()
    except OSError as exc:
        logger.warning("could not read drift HTML :: %s", exc)
        return ""


# ── Refresh handler ──────────────────────────────────────────────────────


def _refresh_all():
    headline = _headline_card()
    health = _pipeline_health_card()
    sal = _salary_distribution_df()
    top_co = _top_companies_df()
    role_share = _role_family_share_df()
    skills = _top_skills_df()
    drift_html_path, drift_metrics = _latest_drift()
    drift_md = _drift_summary_md(drift_metrics)
    drift_html = _drift_html(drift_html_path)
    return headline, health, sal, top_co, role_share, skills, drift_md, drift_html


# ── Gradio tab ────────────────────────────────────────────────────────────


def build_tab() -> gr.Tab:
    with gr.Tab("Dashboard") as tab:
        gr.Markdown(
            "## Operational dashboard\n\n"
            "Pipeline health, market trends, and drift detection over the "
            "curated corpus. Refresh to recompute everything from the "
            "latest parquet on disk."
        )
        refresh_btn = gr.Button("Refresh", variant="primary")

        with gr.Row():
            headline_md = gr.Markdown(_headline_card())
            health_md = gr.Markdown(_pipeline_health_card())

        with gr.Tab("Salary distribution"):
            gr.Markdown("Median + p25/p75 of model-predicted salary, sliced by role × seniority.")
            sal_df = gr.Dataframe(value=_salary_distribution_df(), interactive=False, wrap=True)
        with gr.Tab("Top companies"):
            gr.Markdown("Top 20 companies by open posting count, with role-family breakdown.")
            companies_df = gr.Dataframe(value=_top_companies_df(), interactive=False, wrap=True)
        with gr.Tab("Role family by country"):
            gr.Markdown("Per-country share of each role family.")
            role_df = gr.Dataframe(value=_role_family_share_df(), interactive=False, wrap=True)
        with gr.Tab("Top skills"):
            gr.Markdown("Most-mentioned canonical skills (regex `tech_stack` extractor).")
            skills_df = gr.Dataframe(value=_top_skills_df(), interactive=False, wrap=True)

        gr.Markdown("---")
        # Drift section
        drift_html_path, drift_metrics = _latest_drift()
        drift_md = gr.Markdown(_drift_summary_md(drift_metrics))
        drift_html_box = gr.HTML(value=_drift_html(drift_html_path))

        refresh_btn.click(
            _refresh_all,
            inputs=None,
            outputs=[
                headline_md,
                health_md,
                sal_df,
                companies_df,
                role_df,
                skills_df,
                drift_md,
                drift_html_box,
            ],
        )
    return tab
