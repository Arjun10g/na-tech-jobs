"""Salary prediction tab — paste-a-JD or fill the form, get a back-transformed
USD/year point + range from the Phase 2 Step 3 XGBoost model.
"""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

from app.feature_form import (
    features_from_description,
    humanize_row,
    merge_form_overrides,
)
from app.model_loader import get_predictor

logger = logging.getLogger("app.tabs.salary")

# Backed by the Tier-5 test-set MAE from the model card (~$29k). The "range"
# we display is ±MAE around the point estimate — a rough but honest
# uncertainty band. Phase 4+ may replace this with quantile-regression bands.
DEFAULT_MAE_USD = 29_000


SAMPLE_DESC = """\
We're hiring a Senior Machine Learning Engineer to work on our recommendation
systems. This is a hybrid role, 3 days per week in our San Francisco office.

Requirements:
- 7+ years of experience building production ML systems
- Bachelor's degree in Computer Science or related field
- Strong Python and PyTorch experience
- Familiarity with AWS and Kubernetes

Compensation:
The annual base salary range for this role is $200,000 - $260,000 USD,
plus RSUs and a target performance bonus. We offer relocation assistance.
"""


def _format_usd(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "—"
    return f"${value:,.0f}"


def _predict(features: pd.DataFrame, mae_band: float) -> dict[str, Any]:
    predictor = get_predictor()
    log_pred = predictor.predict_log_usd_yearly(features)
    point = float(10.0 ** log_pred[0])
    return {
        "point": point,
        "low": max(0.0, point - mae_band),
        "high": point + mae_band,
    }


def _predict_from_description(
    description: str,
    title: str,
    country_hint: str,
) -> tuple[str, str, str, str, str]:
    if not description.strip():
        empty = "_paste a job description to predict_"
        return empty, "", "", "", ""
    feats = features_from_description(
        description_md=description,
        title=title or "",
        country_hint=country_hint,
    )
    summary = humanize_row(feats)
    try:
        prediction = _predict(feats, mae_band=DEFAULT_MAE_USD)
    except Exception as exc:  # noqa: BLE001
        logger.exception("salary prediction failed")
        return summary, "", "", "", f"⚠️ prediction failed: {exc}"
    point_md = (
        f"### Predicted salary\n\n"
        f"**{_format_usd(prediction['point'])} / year** (USD)\n\n"
        f"_Range: {_format_usd(prediction['low'])} – "
        f"{_format_usd(prediction['high'])} (point ± model MAE)_\n\n"
        "The model was trained on the disclosed-salary subset of NA tech ATS "
        "postings (Greenhouse / Lever / Ashby) — see [model card]"
        "(https://huggingface.co/arjun10g/na-tech-jobs-salary-v1) "
        "for the honest framing on selection bias."
    )
    return (
        summary,
        str(prediction["point"]),
        str(prediction["low"]),
        str(prediction["high"]),
        point_md,
    )


def _predict_from_form(
    description: str,
    title: str,
    country_hint: str,
    yoe_override: int | None,
    edu_override: str | None,
    seniority_override: str | None,
    role_family_override: str | None,
    remote_override: str | None,
    equity_override: str | None,
) -> str:
    base = features_from_description(description, title, country_hint=country_hint)
    overrides = {
        "min_years_experience": yoe_override,
        "min_education": edu_override,
        "seniority_extracted": seniority_override,
        "role_family_extracted": role_family_override,
        "remote_policy": remote_override,
        "offers_equity": True
        if equity_override == "yes"
        else False
        if equity_override == "no"
        else None,
    }
    merged = merge_form_overrides(base, overrides)
    try:
        prediction = _predict(merged, mae_band=DEFAULT_MAE_USD)
    except Exception as exc:  # noqa: BLE001
        logger.exception("salary prediction failed")
        return f"⚠️ prediction failed: {exc}"
    return (
        f"### With overrides\n\n"
        f"**{_format_usd(prediction['point'])} / year** (USD)\n\n"
        f"_Range: {_format_usd(prediction['low'])} – "
        f"{_format_usd(prediction['high'])}_"
    )


def build_tab() -> gr.Tab:
    with gr.Tab("Salary predictor") as tab:
        gr.Markdown(
            "## Predict NA tech salary\n\n"
            "Paste a job description (markdown or HTML — both work). The "
            "regex cascade extracts ~20 features; the XGBoost regressor "
            "predicts the maximum salary in USD/year. Adjust any extracted "
            "field below to see how the prediction changes.\n\n"
            "_The model is purely tabular (Phase 2). bge-m3 description "
            "embeddings + a higher MAE/MAPE land in Phase 5._"
        )

        with gr.Row():
            with gr.Column(scale=2):
                title_input = gr.Textbox(
                    label="Job title (optional)",
                    placeholder="Senior Machine Learning Engineer",
                    lines=1,
                )
                description_input = gr.Textbox(
                    label="Job description",
                    placeholder="Paste the full description here…",
                    lines=15,
                    value=SAMPLE_DESC,
                )
                country_hint = gr.Radio(
                    label="Country (when location isn't in the JD)",
                    choices=["US", "CA"],
                    value="US",
                )
                predict_btn = gr.Button("Predict salary", variant="primary")
            with gr.Column(scale=1):
                prediction_md = gr.Markdown(
                    "### Predicted salary\n\n_paste a description and click Predict_"
                )
                summary_md = gr.Markdown("### Extracted features\n_(none yet)_")
                state_point = gr.State("")
                state_low = gr.State("")
                state_high = gr.State("")

        gr.Markdown(
            "---\n### Override extracted fields\n\n"
            "_Tweak any of these and click 'Predict with overrides' to "
            "see how the prediction shifts._"
        )

        with gr.Row():
            yoe_override = gr.Slider(
                label="Min years experience",
                minimum=0,
                maximum=20,
                step=1,
                value=5,
            )
            edu_override = gr.Dropdown(
                label="Min education",
                choices=["high_school", "associates", "bachelors", "masters", "phd"],
                value="bachelors",
            )
            seniority_override = gr.Dropdown(
                label="Seniority",
                choices=[
                    "intern",
                    "junior",
                    "mid",
                    "senior",
                    "staff",
                    "principal",
                    "manager",
                    "director",
                    "exec",
                ],
                value="senior",
            )
        with gr.Row():
            role_family_override = gr.Dropdown(
                label="Role family",
                choices=["DS", "DA", "DE", "MLE", "RS", "AS", "SWE-ML", "Manager", "Other"],
                value="MLE",
            )
            remote_override = gr.Dropdown(
                label="Remote policy",
                choices=["onsite", "hybrid", "remote", "remote-na"],
                value="hybrid",
            )
            equity_override = gr.Radio(
                label="Equity?",
                choices=["yes", "no"],
                value="yes",
            )

        override_btn = gr.Button("Predict with overrides")
        override_md = gr.Markdown("")

        predict_btn.click(
            fn=_predict_from_description,
            inputs=[description_input, title_input, country_hint],
            outputs=[summary_md, state_point, state_low, state_high, prediction_md],
        )
        override_btn.click(
            fn=_predict_from_form,
            inputs=[
                description_input,
                title_input,
                country_hint,
                yoe_override,
                edu_override,
                seniority_override,
                role_family_override,
                remote_override,
                equity_override,
            ],
            outputs=override_md,
        )

    return tab
