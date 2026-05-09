"""Market-trend rollups for the dashboard tab.

Pure read-only summaries computed on demand from the latest curated
parquet — no caching, no precomputation. The numbers are small enough
(~12k rows) that DuckDB chews through them in <100 ms.

Each helper returns a tidy DataFrame the dashboard renders directly,
plus a compact JSON-friendly view for tests / API consumers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("monitoring.market_trends")

DEFAULT_CURATED = Path("data/curated_enriched/jobs.parquet")
FALLBACK_CURATED = Path("data/curated/jobs.parquet")


def _resolve(path: Path | None) -> Path:
    if path is not None:
        return path
    return DEFAULT_CURATED if DEFAULT_CURATED.exists() else FALLBACK_CURATED


def _load(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(_resolve(path))


# ── Salary distribution ───────────────────────────────────────────────────


def salary_distribution(
    df: pd.DataFrame | None = None,
    *,
    role_col: str = "role_family_v1",
    seniority_col: str = "seniority_label_v1",
    salary_col: str = "predicted_salary_usd_v1",
) -> pd.DataFrame:
    """Median + p25/p75 of predicted salary, sliced by role × seniority.

    Falls back to extracted columns + (min+max)/2 disclosed salary when
    Phase 4 versioned columns aren't on the parquet.
    """
    if df is None:
        df = _load()
    if role_col not in df.columns:
        role_col = "role_family_extracted"
    if seniority_col not in df.columns:
        seniority_col = "seniority_extracted"
    if salary_col not in df.columns:
        # Synthesize from disclosed.
        df = df.assign(
            _salary=(
                df.get("salary_min_usd_yearly", pd.Series(dtype="float"))
                + df.get("salary_max_usd_yearly", pd.Series(dtype="float"))
            )
            / 2.0
        )
        salary_col = "_salary"

    g = df.groupby([role_col, seniority_col], dropna=True)[salary_col]
    out = (
        g.agg(
            n="count",
            p25=lambda x: float(x.quantile(0.25)),
            median=lambda x: float(x.quantile(0.50)),
            p75=lambda x: float(x.quantile(0.75)),
        )
        .reset_index()
        .rename(columns={role_col: "role_family", seniority_col: "seniority"})
        .sort_values(["role_family", "seniority"])
    )
    return out


# ── Top companies ────────────────────────────────────────────────────────


def top_companies(
    df: pd.DataFrame | None = None,
    *,
    limit: int = 25,
    role_col: str = "role_family_v1",
) -> pd.DataFrame:
    """Top employers by total open postings, with role-family breakdown."""
    if df is None:
        df = _load()
    if role_col not in df.columns:
        role_col = "role_family_extracted"
    base = df.groupby("company_name").size().rename("n_total")
    top = base.sort_values(ascending=False).head(limit).index
    df_top = df[df["company_name"].isin(top)]
    pivot = df_top.groupby(["company_name", role_col]).size().unstack(fill_value=0)
    pivot["n_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("n_total", ascending=False)
    return pivot.reset_index()


# ── Role-family proportions ──────────────────────────────────────────────


def role_family_share(
    df: pd.DataFrame | None = None,
    *,
    role_col: str = "role_family_v1",
) -> pd.DataFrame:
    """Per-country share of each role family (% of jobs)."""
    if df is None:
        df = _load()
    if role_col not in df.columns:
        role_col = "role_family_extracted"
    counts = df.groupby(["country", role_col]).size().rename("n").reset_index()
    totals = df.groupby("country").size().rename("n_country").reset_index()
    out = counts.merge(totals, on="country")
    out["share_pct"] = (out["n"] / out["n_country"] * 100).round(1)
    out = out.rename(columns={role_col: "role_family"})
    return out.sort_values(["country", "n"], ascending=[True, False])


# ── Top skills (regex tech_stack) ─────────────────────────────────────────


def top_skills(
    df: pd.DataFrame | None = None,
    *,
    limit: int = 30,
    skill_col: str = "extracted_skills_v1",
) -> pd.DataFrame:
    """Most-mentioned skills across the corpus."""
    if df is None:
        df = _load()
    if skill_col not in df.columns:
        skill_col = "tech_stack"
    if skill_col not in df.columns:
        return pd.DataFrame(columns=["skill", "n_jobs"])

    # Each row's value is a list / numpy array of canonical skill names.
    counts: dict[str, int] = {}
    for v in df[skill_col].tolist():
        if v is None:
            continue
        try:
            iterable = list(v)
        except TypeError:
            continue
        for s in iterable:
            if s is None:
                continue
            s = str(s)
            counts[s] = counts.get(s, 0) + 1
    out = (
        pd.DataFrame([{"skill": k, "n_jobs": v} for k, v in counts.items()])
        .sort_values("n_jobs", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
    return out


# ── Headline numbers (for the dashboard top card) ────────────────────────


def headline_numbers(df: pd.DataFrame | None = None) -> dict:
    """Single-card stats — 12k jobs at a glance."""
    if df is None:
        df = _load()
    n = len(df)
    n_disclosed = (
        int(df["salary_disclosed"].fillna(False).sum()) if "salary_disclosed" in df.columns else 0
    )
    n_us = int((df["country"] == "US").sum()) if "country" in df.columns else 0
    n_ca = int((df["country"] == "CA").sum()) if "country" in df.columns else 0
    n_companies = int(df["company_name"].nunique()) if "company_name" in df.columns else 0
    median_disclosed_salary: float | None = None
    if "salary_disclosed" in df.columns and "salary_max_usd_yearly" in df.columns:
        d = df.loc[df["salary_disclosed"] == True, "salary_max_usd_yearly"].dropna()  # noqa: E712
        if len(d) > 0:
            median_disclosed_salary = float(d.median())
    median_predicted_salary: float | None = None
    if "predicted_salary_usd_v1" in df.columns:
        d = df["predicted_salary_usd_v1"].dropna()
        if len(d) > 0:
            median_predicted_salary = float(d.median())
    return {
        "n_jobs_active": n,
        "n_companies": n_companies,
        "n_us": n_us,
        "n_ca": n_ca,
        "n_salary_disclosed": n_disclosed,
        "salary_disclosure_rate": round(n_disclosed / max(n, 1), 3),
        "median_disclosed_salary_usd": (
            round(median_disclosed_salary) if median_disclosed_salary else None
        ),
        "median_predicted_salary_usd": (
            round(median_predicted_salary) if median_predicted_salary else None
        ),
    }
