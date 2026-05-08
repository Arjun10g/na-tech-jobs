"""Tests for ingestion.normalize."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ingestion.normalize import (
    extract_role_family,
    extract_seniority,
    is_likely_french,
    normalize,
    normalize_salary,
    parse_location,
)
from ingestion.schema import CanonicalJob


def make_job(**overrides) -> CanonicalJob:
    base = dict(
        id="abcdef0123456789",
        company_slug="acme",
        company_name="Acme",
        title="Senior Data Scientist",
        url="https://acme.com/jobs/1",
        location_raw="San Francisco, CA",
        scraped_at=datetime.now(timezone.utc),
        source="greenhouse",
        raw_payload_hash="deadbeef",
    )
    base.update(overrides)
    return CanonicalJob(**base)


@pytest.mark.parametrize(
    "raw,expected_country,expected_region",
    [
        ("San Francisco, CA", "US", "CA"),
        ("New York, NY, United States", "US", "NY"),
        ("Toronto, ON, Canada", "CA", "ON"),
        ("Vancouver, British Columbia", "CA", "BC"),
        ("Montréal, QC", "CA", "QC"),
        ("Remote - US", "US", None),
        ("London, UK", None, None),
    ],
)
def test_parse_location_country_and_region(raw, expected_country, expected_region):
    out = parse_location(raw)
    assert out["country"] == expected_country
    assert out["region"] == expected_region


def test_parse_location_remote_in_na_becomes_remote_na():
    out = parse_location("Remote - US")
    assert out["remote_policy"] == "remote-na"


def test_parse_location_hybrid_signal():
    out = parse_location("Hybrid - New York, NY")
    assert out["remote_policy"] == "hybrid"


def test_parse_location_default_country_fallback():
    out = parse_location("Remote", default_country="CA")
    assert out["country"] == "CA"
    assert out["remote_policy"] == "remote-na"


def test_parse_location_empty_returns_default():
    out = parse_location(None, default_country="US")
    assert out == {"country": "US", "region": None, "city": None, "remote_policy": None}


def test_normalize_salary_usd_year_passthrough():
    lo, hi = normalize_salary(100_000, 200_000, "USD", "year")
    assert (lo, hi) == (100_000, 200_000)


def test_normalize_salary_cad_to_usd():
    lo, hi = normalize_salary(130_000, 195_000, "CAD", "year")
    assert lo == pytest.approx(130_000 * 0.73, rel=1e-6)
    assert hi == pytest.approx(195_000 * 0.73, rel=1e-6)


def test_normalize_salary_hourly_to_yearly():
    lo, hi = normalize_salary(50, 75, "USD", "hour")
    assert lo == pytest.approx(50 * 2080)
    assert hi == pytest.approx(75 * 2080)


def test_normalize_salary_unknown_currency_returns_none():
    assert normalize_salary(100, 200, "EUR", "year") == (None, None)


def test_normalize_salary_no_inputs_returns_none():
    assert normalize_salary(None, None, "USD", "year") == (None, None)


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Senior Software Engineer", "senior"),
        ("Sr. Data Scientist", "senior"),
        ("Junior Backend Engineer", "junior"),
        ("Staff Machine Learning Engineer", "staff"),
        ("Principal Research Scientist", "principal"),
        ("Engineering Manager", "manager"),
        ("Director of Data Science", "director"),
        ("Software Engineering Intern", "intern"),
        ("Data Scientist", "mid"),  # default fallback
    ],
)
def test_extract_seniority(title, expected):
    assert extract_seniority(title) == expected


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Machine Learning Engineer", "MLE"),
        ("Senior Data Scientist", "DS"),
        ("Data Engineer", "DE"),
        ("Business Analyst", "DA"),
        ("Research Scientist", "RS"),
        ("AI Engineer, LLMs", "SWE-ML"),
        ("Director of Engineering", "Manager"),
        ("Frontend Engineer", "Other"),
    ],
)
def test_extract_role_family(title, expected):
    assert extract_role_family(title) == expected


def test_normalize_full_pipeline_us_role():
    job = make_job(
        title="Senior Machine Learning Engineer (Remote)",
        location_raw="San Francisco, CA",
        salary_min=180_000,
        salary_max=240_000,
        salary_currency="USD",
        salary_period="year",
    )
    out = normalize(job)
    assert out.title == "Senior Machine Learning Engineer"
    assert out.country == "US"
    assert out.region == "CA"
    assert out.seniority_extracted == "senior"
    assert out.role_family_extracted == "MLE"
    assert out.salary_min_usd_yearly == pytest.approx(180_000)
    assert out.salary_max_usd_yearly == pytest.approx(240_000)


def test_normalize_falls_back_to_default_country_for_remote():
    job = make_job(title="Data Engineer", location_raw="Remote")
    out = normalize(job, default_country="CA")
    assert out.country == "CA"
    assert out.remote_policy == "remote-na"


def test_is_likely_french_title():
    job = make_job(title="Ingénieur en apprentissage automatique", location_raw="Montréal, QC")
    assert is_likely_french(job)


def test_is_likely_french_english_in_quebec_kept():
    job = make_job(
        title="Senior Software Engineer",
        location_raw="Montréal, QC",
        description_md="We are hiring engineers to build distributed systems.",
    )
    assert not is_likely_french(job)
