"""Tests for monitoring.market_trends — pure-pandas computations."""

from __future__ import annotations

import pandas as pd

from monitoring.market_trends import (
    headline_numbers,
    role_family_share,
    salary_distribution,
    top_companies,
    top_skills,
)


def _df():
    return pd.DataFrame(
        [
            {
                "id": "j1",
                "title": "Senior MLE",
                "company_name": "Acme",
                "country": "US",
                "salary_disclosed": True,
                "salary_min_usd_yearly": 180_000.0,
                "salary_max_usd_yearly": 240_000.0,
                "seniority_label_v1": "senior",
                "role_family_v1": "MLE",
                "predicted_salary_usd_v1": 210_000.0,
                "extracted_skills_v1": ["Python", "PyTorch"],
            },
            {
                "id": "j2",
                "title": "Senior MLE",
                "company_name": "Acme",
                "country": "US",
                "salary_disclosed": False,
                "salary_min_usd_yearly": None,
                "salary_max_usd_yearly": None,
                "seniority_label_v1": "senior",
                "role_family_v1": "MLE",
                "predicted_salary_usd_v1": 220_000.0,
                "extracted_skills_v1": ["Python"],
            },
            {
                "id": "j3",
                "title": "Staff DE",
                "company_name": "Beta",
                "country": "CA",
                "salary_disclosed": True,
                "salary_min_usd_yearly": 200_000.0,
                "salary_max_usd_yearly": 260_000.0,
                "seniority_label_v1": "staff",
                "role_family_v1": "DE",
                "predicted_salary_usd_v1": 230_000.0,
                "extracted_skills_v1": ["dbt", "Snowflake"],
            },
            {
                "id": "j4",
                "title": "DS",
                "company_name": "Beta",
                "country": "CA",
                "salary_disclosed": True,
                "salary_min_usd_yearly": 130_000.0,
                "salary_max_usd_yearly": 170_000.0,
                "seniority_label_v1": "senior",
                "role_family_v1": "DS",
                "predicted_salary_usd_v1": 150_000.0,
                "extracted_skills_v1": ["SQL", "Python"],
            },
        ]
    )


# ── headline_numbers ──────────────────────────────────────────────────────


def test_headline_numbers_basic_counts():
    h = headline_numbers(_df())
    assert h["n_jobs_active"] == 4
    assert h["n_companies"] == 2
    assert h["n_us"] == 2
    assert h["n_ca"] == 2
    assert h["n_salary_disclosed"] == 3
    assert h["salary_disclosure_rate"] == 0.75


def test_headline_numbers_medians():
    h = headline_numbers(_df())
    # Disclosed max salaries: 240k, 260k, 170k → median 240k.
    assert h["median_disclosed_salary_usd"] == 240_000
    # Predicted: 210, 220, 230, 150 → median = 215.
    assert h["median_predicted_salary_usd"] == 215_000


# ── salary_distribution ───────────────────────────────────────────────────


def test_salary_distribution_returns_one_row_per_role_seniority():
    out = salary_distribution(_df())
    keys = {(r.role_family, r.seniority) for r in out.itertuples()}
    assert ("MLE", "senior") in keys
    assert ("DE", "staff") in keys
    assert ("DS", "senior") in keys


def test_salary_distribution_aggregates_correctly():
    out = salary_distribution(_df())
    mle = out[(out["role_family"] == "MLE") & (out["seniority"] == "senior")]
    assert int(mle["n"].iloc[0]) == 2
    # Median of [210, 220] = 215
    assert int(mle["median"].iloc[0]) == 215_000


# ── top_companies ─────────────────────────────────────────────────────────


def test_top_companies_orders_by_total_count():
    out = top_companies(_df(), limit=10)
    assert "n_total" in out.columns
    assert list(out["company_name"][:2]) == ["Acme", "Beta"] or list(out["company_name"][:2]) == [
        "Beta",
        "Acme",
    ]


def test_top_companies_pivot_has_role_columns():
    out = top_companies(_df(), limit=10)
    # Acme has 2 MLE; Beta has 1 DE + 1 DS.
    acme = out[out["company_name"] == "Acme"].iloc[0]
    assert acme.get("MLE", 0) == 2
    beta = out[out["company_name"] == "Beta"].iloc[0]
    assert beta.get("DE", 0) == 1
    assert beta.get("DS", 0) == 1


# ── role_family_share ─────────────────────────────────────────────────────


def test_role_family_share_sums_to_100_per_country():
    out = role_family_share(_df())
    for _country, group in out.groupby("country"):
        assert abs(group["share_pct"].sum() - 100.0) < 0.5


# ── top_skills ────────────────────────────────────────────────────────────


def test_top_skills_counts_correctly():
    out = top_skills(_df(), limit=10)
    skill_to_n = dict(zip(out["skill"], out["n_jobs"], strict=True))
    assert skill_to_n["Python"] == 3  # j1, j2, j4
    assert skill_to_n["PyTorch"] == 1


def test_top_skills_handles_missing_column():
    out = top_skills(_df().drop(columns=["extracted_skills_v1"]), limit=10)
    # Falls back to tech_stack, which is also missing → returns empty.
    assert out.empty or len(out) == 0


def test_top_skills_handles_none_rows():
    df = _df().copy()
    df.at[0, "extracted_skills_v1"] = None
    out = top_skills(df, limit=10)
    skill_to_n = dict(zip(out["skill"], out["n_jobs"], strict=True))
    # Python originally appeared 3x; with j1 nulled → 2.
    assert skill_to_n["Python"] == 2
