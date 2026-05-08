"""Pandera-based quality validation for the canonical job DataFrame.

Failures don't crash the orchestrator immediately; instead they're rolled up
into a structured summary so a single bad row doesn't take down the snapshot.
The summary is written to `reports/quality/<date>.json` on the Dataset repo
and posted to Discord.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.errors as pe

from ingestion.schema import CANONICAL_COLUMNS, JOB_SCHEMA


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure dtypes line up with the schema before validation.

    Ensures required columns exist (filled with NaN if missing), trims to the
    canonical column set, and coerces datetime columns to UTC.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    for c in missing:
        df[c] = None

    extras = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    df = df.drop(columns=extras)
    df = df[list(CANONICAL_COLUMNS)]

    for ts_col in ("posted_at", "scraped_at"):
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if "salary_disclosed" in df.columns:
        df["salary_disclosed"] = df["salary_disclosed"].fillna(False).astype(bool)
    if "extraction_version" in df.columns:
        df["extraction_version"] = df["extraction_version"].fillna("v1").astype(str)

    string_cols = [
        # identity / location / source
        "id",
        "company_slug",
        "company_name",
        "title",
        "url",
        "location_raw",
        "country",
        "region",
        "city",
        "remote_policy",
        "seniority_extracted",
        "role_family_extracted",
        "salary_currency",
        "salary_period",
        "description_md",
        "source",
        "raw_payload_hash",
        # Phase 2 enum-as-string columns (Pandera enforces the value set)
        "min_education",
        "clearance_level",
        "offers_visa_sponsorship",
        "equity_form",
        "bonus_type",
        "contract_type",
        "manager_role",
        "posting_quality",
        "team_or_department",
        "extraction_version",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), None)
            df[col] = df[col].astype("object")

    float_cols = ["salary_min", "salary_max", "salary_min_usd_yearly", "salary_max_usd_yearly"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Nullable integers (pandas Int64 allows NaN-as-NA, unlike numpy int64).
    nullable_int_cols = [
        "min_years_experience",
        "max_years_experience",
        "max_travel_percent",
        "direct_reports_count",
    ]
    for col in nullable_int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Nullable booleans roundtripped through pandas BooleanDtype.
    nullable_bool_cols = [
        "requires_security_clearance",
        "offers_relocation",
        "offers_equity",
        "bonus_mentioned",
        "on_call_required",
    ]
    for col in nullable_bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    # List/dict object columns — leave as-is; parquet handles them natively.
    object_cols = [
        "requires_citizenship",
        "language_requirements",
        "tech_stack",
        "industry_experience",
        "extraction_meta",
    ]
    for col in object_cols:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), None)

    return df


def validate(df: pd.DataFrame, *, strict: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the canonical schema. Returns (clean_df, report).

    With ``strict=False`` (default), failing rows are dropped from the output
    and counted in the report so one bad posting doesn't fail the snapshot.
    """
    df = coerce_dtypes(df)
    report: dict[str, Any] = {
        "input_rows": len(df),
        "valid_rows": 0,
        "dropped_rows": 0,
        "errors": [],
    }

    try:
        validated = JOB_SCHEMA.validate(df, lazy=True)
        report["valid_rows"] = len(validated)
        return validated, report
    except pe.SchemaErrors as exc:
        failure_index = set()
        if exc.failure_cases is not None and "index" in exc.failure_cases:
            failure_index = set(exc.failure_cases["index"].dropna().astype(int).tolist())
        report["errors"] = (
            exc.failure_cases.head(20).to_dict(orient="records")
            if exc.failure_cases is not None
            else [str(exc)]
        )
        if strict:
            raise
        clean = df.drop(index=list(failure_index), errors="ignore").reset_index(drop=True)
        try:
            clean = JOB_SCHEMA.validate(clean, lazy=True)
        except pe.SchemaErrors as exc2:
            # second pass produced new failures — fall back to dropping them silently
            still_bad = set()
            if exc2.failure_cases is not None and "index" in exc2.failure_cases:
                still_bad = set(exc2.failure_cases["index"].dropna().astype(int).tolist())
            clean = clean.drop(index=list(still_bad), errors="ignore").reset_index(drop=True)
            report["errors"].extend(
                exc2.failure_cases.head(20).to_dict(orient="records")
                if exc2.failure_cases is not None
                else []
            )
        report["valid_rows"] = len(clean)
        report["dropped_rows"] = report["input_rows"] - report["valid_rows"]
        return clean, report
