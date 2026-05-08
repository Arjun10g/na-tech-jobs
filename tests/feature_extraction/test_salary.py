"""Tests for ingestion.feature_extraction.regex.salary."""

from __future__ import annotations

import pytest

from ingestion.feature_extraction.regex import salary


def _run(text):
    return salary.run(text)


def test_typical_us_yearly_range():
    out = _run(
        "We expect to pay this role an annual salary of $135,000 - $180,000 USD plus equity."
    )
    assert out["salary_min"].value == 135_000
    assert out["salary_max"].value == 180_000
    assert out["salary_currency"].value == "USD"
    assert out["salary_period"].value == "year"
    assert out["salary_disclosed"].value is True


def test_k_suffix_range():
    out = _run("Compensation: $135K - $180K annually.")
    assert out["salary_min"].value == 135_000
    assert out["salary_max"].value == 180_000
    assert out["salary_period"].value == "year"


def test_cad_range():
    out = _run("The base salary range for this role is CAD 130,000 to 175,000 per year.")
    assert out["salary_min"].value == 130_000
    assert out["salary_max"].value == 175_000
    assert out["salary_currency"].value == "CAD"


def test_hourly_range():
    out = _run("Pay range: $22.00 - $29.00 per hour. This is an hourly role.")
    assert out["salary_min"].value == 22.0
    assert out["salary_max"].value == 29.0
    assert out["salary_period"].value == "hour"


def test_no_anchor_keyword_skipped():
    """Without an anchor word, raw $X-$Y patterns are ignored."""
    out = _run("We raised $135M - $180M Series C from top investors.")
    assert out == {}


def test_implausible_low_yearly_skipped():
    out = _run("Salary range: $5,000 - $8,000 annually.")
    assert out == {}


def test_gift_card_skipped():
    out = _run("Get a $50 gift card when you refer a friend.")
    assert out == {}


def test_only_one_value_skipped():
    """Single $X without a range is not enough to call disclosed."""
    out = _run("Base salary of $150,000.")
    assert out == {}


@pytest.mark.parametrize(
    "snippet,expected_min,expected_max",
    [
        ("Annual base salary range: $90,000-$110,000 USD.", 90_000, 110_000),
        ("Pay range is USD 90,000 to 110,000.", 90_000, 110_000),
        ("compensation: $90K-$110K", 90_000, 110_000),
        ("Salary range $90,000 – $110,000 (USD).", 90_000, 110_000),
    ],
)
def test_format_variants(snippet, expected_min, expected_max):
    out = _run(snippet)
    assert out["salary_min"].value == expected_min
    assert out["salary_max"].value == expected_max
