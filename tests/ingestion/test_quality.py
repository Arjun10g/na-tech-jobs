"""Tests for ingestion.quality."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ingestion.dedup import to_dataframe
from ingestion.quality import validate
from ingestion.schema import CanonicalJob


def make_job(idx: int, **overrides) -> CanonicalJob:
    base = dict(
        id=f"{idx:016x}",
        company_slug="acme",
        company_name="Acme",
        title=f"Senior Engineer {idx}",
        url=f"https://acme.com/jobs/{idx}",
        country="US",
        region="CA",
        city="San Francisco",
        remote_policy="hybrid",
        salary_disclosed=False,
        description_md="A description.",
        scraped_at=datetime.now(timezone.utc),
        source="greenhouse",
        raw_payload_hash="deadbeefdeadbeef",
    )
    base.update(overrides)
    return CanonicalJob(**base)


def test_validate_clean_dataframe():
    jobs = [make_job(i) for i in range(3)]
    df = to_dataframe(jobs)
    valid, report = validate(df)
    assert report["valid_rows"] == 3
    assert report["dropped_rows"] == 0
    assert len(valid) == 3


def test_validate_drops_rows_with_invalid_country():
    jobs = [make_job(1), make_job(2, country="UK")]
    df = to_dataframe(jobs)
    valid, report = validate(df)
    assert report["dropped_rows"] >= 1
    assert "UK" not in valid["country"].dropna().tolist()


def test_validate_handles_missing_optional_columns():
    jobs = [make_job(1)]
    df = to_dataframe(jobs).drop(columns=["region"])  # quality.coerce_dtypes refills
    valid, report = validate(df)
    assert "region" in valid.columns
    assert report["valid_rows"] == 1


def test_validate_rejects_duplicate_ids():
    jobs = [make_job(1), make_job(1)]  # same id
    df = to_dataframe(jobs)
    valid, report = validate(df)
    assert len(valid) <= 1
    assert report["dropped_rows"] >= 1


def test_validate_coerces_timestamp_strings():
    jobs = [make_job(1)]
    df = to_dataframe(jobs)
    df["scraped_at"] = df["scraped_at"].astype(str)
    valid, _ = validate(df)
    assert pd.api.types.is_datetime64_any_dtype(valid["scraped_at"])
