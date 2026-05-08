"""Map a pasted job description (or manual form fields) to a single-row
DataFrame the salary predictor can consume.

The cascade in ``ingestion.feature_extraction`` already extracts most of the
features the regressor expects; this helper just bridges the gap between
"user pastes a JD" and "predictor consumes a 24-column DataFrame row".
"""

from __future__ import annotations

import html
import logging
from datetime import timezone
from typing import Any

import pandas as pd
from markdownify import markdownify

logger = logging.getLogger("app.feature_form")

# All 24 input columns the salary regressor was trained on (per
# ``models/salary/dataset.py``). Anything missing here defaults to None and
# the encoder/imputer handles it.
PREDICTOR_COLUMNS: tuple[str, ...] = (
    # continuous
    "min_years_experience",
    # ordinal
    "min_education",
    "seniority_extracted",
    "manager_role",
    "clearance_level",
    # low-card nominal
    "country",
    "source",
    "role_family_extracted",
    "remote_policy",
    "contract_type",
    "equity_form",
    "bonus_type",
    # high-card nominal
    "region",
    "city",
    # boolean / tri-state
    "requires_security_clearance",
    "offers_visa_sponsorship",
    "offers_relocation",
    "offers_equity",
    "bonus_mentioned",
    "on_call_required",
    # lists
    "requires_citizenship",
    "language_requirements",
    "tech_stack",
    # datetime
    "posted_at",
)


def _description_to_markdown(text: str) -> str:
    """Mirror the ingestion pipeline's HTML→MD step so the cascade sees
    clean markdown regardless of whether the user pastes HTML or text."""
    if not text:
        return ""
    if "<" in text and ">" in text:
        return markdownify(html.unescape(text), heading_style="ATX").strip()
    return text


def features_from_description(
    description_md: str,
    title: str = "",
    *,
    country_hint: str | None = "US",
) -> pd.DataFrame:
    """Run the regex cascade over a description and return a single-row
    DataFrame ready for the salary predictor.

    ``country_hint`` is a fallback when the cascade can't detect country
    (most pasted descriptions don't include "United States" verbatim).
    """
    from ingestion.feature_extraction import extract_features
    from ingestion.normalize import (
        extract_role_family,
        extract_seniority,
        parse_location,
    )

    desc = _description_to_markdown(description_md)
    title = title.strip()
    cascade = extract_features(desc, title=title, use_llm=False)

    loc = parse_location(None, default_country=country_hint)

    row: dict[str, Any] = dict.fromkeys(PREDICTOR_COLUMNS, None)
    row.update(
        {
            "min_years_experience": cascade.get("min_years_experience"),
            "min_education": cascade.get("min_education"),
            "seniority_extracted": extract_seniority(title) if title else None,
            "role_family_extracted": extract_role_family(title) if title else None,
            "manager_role": cascade.get("manager_role"),
            "clearance_level": cascade.get("clearance_level"),
            "country": loc["country"] or country_hint,
            "source": "greenhouse",  # arbitrary default; the regressor uses it weakly
            "remote_policy": cascade.get("remote_policy_extracted"),
            "contract_type": cascade.get("contract_type"),
            "equity_form": cascade.get("equity_form"),
            "bonus_type": cascade.get("bonus_type"),
            "region": loc["region"],
            "city": loc["city"],
            "requires_security_clearance": cascade.get("requires_security_clearance"),
            "offers_visa_sponsorship": cascade.get("offers_visa_sponsorship"),
            "offers_relocation": cascade.get("offers_relocation"),
            "offers_equity": cascade.get("offers_equity"),
            "bonus_mentioned": cascade.get("bonus_mentioned"),
            "on_call_required": cascade.get("on_call_required"),
            "requires_citizenship": cascade.get("requires_citizenship"),
            "language_requirements": cascade.get("language_requirements"),
            "tech_stack": cascade.get("tech_stack"),
            "posted_at": pd.Timestamp.now(tz=timezone.utc),
        }
    )
    return pd.DataFrame([row], columns=list(PREDICTOR_COLUMNS))


def features_from_form(form: dict[str, Any]) -> pd.DataFrame:
    """Same shape as ``features_from_description`` but driven by manual UI
    form values. Caller passes a dict of field → value; missing fields
    default to None."""
    row: dict[str, Any] = dict.fromkeys(PREDICTOR_COLUMNS, None)
    row.update({k: v for k, v in form.items() if k in PREDICTOR_COLUMNS})
    if row.get("posted_at") is None:
        row["posted_at"] = pd.Timestamp.now(tz=timezone.utc)
    return pd.DataFrame([row], columns=list(PREDICTOR_COLUMNS))


def merge_form_overrides(
    base: pd.DataFrame,
    overrides: dict[str, Any],
) -> pd.DataFrame:
    """Apply user-supplied overrides on top of cascade-extracted features."""
    out = base.copy()
    for k, v in overrides.items():
        if k in out.columns and v not in (None, "", "—"):
            out.at[out.index[0], k] = v
    return out


def humanize_row(row: pd.DataFrame) -> str:
    """Return a markdown summary of the parsed features for the UI's
    'what we extracted' panel."""
    if row.empty:
        return "_(no features extracted)_"
    r = row.iloc[0]
    pieces: list[str] = []
    pairs = [
        ("Years of experience", r.get("min_years_experience")),
        ("Minimum education", r.get("min_education")),
        ("Seniority", r.get("seniority_extracted")),
        ("Role family", r.get("role_family_extracted")),
        ("Manager track", r.get("manager_role")),
        ("Country", r.get("country")),
        ("Region", r.get("region")),
        ("City", r.get("city")),
        ("Remote policy", r.get("remote_policy")),
        ("Contract type", r.get("contract_type")),
        ("Offers equity?", r.get("offers_equity")),
        ("Equity form", r.get("equity_form")),
        ("Bonus mentioned?", r.get("bonus_mentioned")),
        ("Bonus type", r.get("bonus_type")),
        ("Requires clearance?", r.get("requires_security_clearance")),
        ("Clearance level", r.get("clearance_level")),
        ("Sponsorship", r.get("offers_visa_sponsorship")),
        ("Tech stack", r.get("tech_stack")),
    ]
    for name, value in pairs:
        if value is None or (isinstance(value, list) and not value):
            continue
        pieces.append(f"- **{name}**: {value}")
    return "\n".join(pieces) if pieces else "_(no features extracted)_"
