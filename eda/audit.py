"""End-to-end statistical / exploratory audit of the curated dataset.

Produces ``data/eda/report.md``, ``data/eda/metrics.json`` and a directory of
PNG plots. Designed to be deterministic, dependency-light (matplotlib +
seaborn + statsmodels), and runnable in one ``uv run python -m eda.audit``.

Sections covered (in this order):
1. Schema + dtype classification (continuous / discrete / ordinal / nominal /
   boolean / list / text / datetime / metadata).
2. Univariate distributions for every predictor.
3. Missingness analysis: per-column rates, co-missing matrix, MCAR test.
4. Target deep-dive: salary_max_usd_yearly raw + log distributions, stratified
   by country / source / role / seniority.
5. Bivariate target vs predictors: continuous (correlation), categorical
   (ANOVA F-test), boolean (Welch t-test), with plots.
6. Multicollinearity: correlation heatmap, VIF, condition number.
7. Outliers: IQR + z-score on salary; contextual sanity checks.
8. Transformation recommendations.
9. Selection-bias / MNAR / omitted-variable discussion specific to salary
   disclosure in the NA tech corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eda.report import write_report

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("eda.audit")

# ── Predictor taxonomy ─────────────────────────────────────────────────────
# We hard-code the role of each column rather than inferring from dtype, since
# many semantically-different columns share dtypes (e.g. country and city are
# both strings but only country has a small bounded vocabulary). This is the
# single source of truth referenced throughout the audit.

TARGET_COL: str = "salary_max_usd_yearly"

CONTINUOUS_PREDICTORS: tuple[str, ...] = (
    "min_years_experience",
    "max_years_experience",
    "max_travel_percent",
    "direct_reports_count",
    "salary_min_usd_yearly",  # informational only — not an input to the regressor
)
ORDINAL_PREDICTORS: dict[str, list[str]] = {
    "min_education": ["high_school", "associates", "bachelors", "masters", "phd"],
    "seniority_extracted": [
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
    "manager_role": ["ic", "tech_lead", "manager", "senior_manager", "director", "exec"],
    "clearance_level": ["public_trust", "confidential", "secret", "top_secret", "ts_sci"],
}
NOMINAL_PREDICTORS: tuple[str, ...] = (
    "country",
    "region",
    "city",
    "source",
    "role_family_extracted",
    "remote_policy",
    "contract_type",
    "equity_form",
    "bonus_type",
    "posting_quality",
    "salary_currency",
    "salary_period",
    "company_slug",
)
BOOLEAN_PREDICTORS: tuple[str, ...] = (
    "salary_disclosed",
    "requires_security_clearance",
    "offers_visa_sponsorship",  # tri-state in practice; treat as nominal-with-3
    "offers_relocation",
    "offers_equity",
    "bonus_mentioned",
    "on_call_required",
)
LIST_PREDICTORS: tuple[str, ...] = (
    "requires_citizenship",
    "language_requirements",
    "tech_stack",
    "industry_experience",
)
TEXT_PREDICTORS: tuple[str, ...] = (
    "title",
    "description_md",
    "team_or_department",
)
DATETIME_COLUMNS: tuple[str, ...] = (
    "posted_at",
    "scraped_at",
    "first_seen_at",
    "last_seen_at",
)
METADATA_COLUMNS: tuple[str, ...] = (
    "id",
    "url",
    "raw_payload_hash",
    "extraction_meta",
    "extraction_version",
    "company_name",
    "location_raw",
    "salary_min",
    "salary_max",
    "times_seen",
)

ALL_PREDICTORS: tuple[str, ...] = (
    CONTINUOUS_PREDICTORS
    + tuple(ORDINAL_PREDICTORS.keys())
    + NOMINAL_PREDICTORS
    + BOOLEAN_PREDICTORS
)


# ── Plot styling ──────────────────────────────────────────────────────────


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titleweight"] = "bold"


def _save_fig(fig: plt.Figure, plots_dir: Path, name: str) -> Path:
    path = plots_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── 1. Schema + dtype classification ──────────────────────────────────────


def audit_schema(df: pd.DataFrame) -> dict[str, Any]:
    n_rows = len(df)
    rows = []
    for col in df.columns:
        if col == TARGET_COL:
            role = "target"
        elif col in CONTINUOUS_PREDICTORS:
            role = "continuous"
        elif col in ORDINAL_PREDICTORS:
            role = "ordinal"
        elif col in NOMINAL_PREDICTORS:
            role = "nominal"
        elif col in BOOLEAN_PREDICTORS:
            role = "boolean"
        elif col in LIST_PREDICTORS:
            role = "list"
        elif col in TEXT_PREDICTORS:
            role = "text"
        elif col in DATETIME_COLUMNS:
            role = "datetime"
        else:
            role = "metadata"
        n_non_null = int(df[col].notna().sum())
        try:
            n_unique = int(df[col].nunique(dropna=True))
        except TypeError:
            # list/dict-valued columns aren't hashable; estimate via str
            n_unique = int(df[col].astype(str).nunique(dropna=True))
        rows.append(
            {
                "column": col,
                "role": role,
                "dtype": str(df[col].dtype),
                "fill_rate": round(n_non_null / n_rows, 4) if n_rows else 0.0,
                "n_unique": n_unique,
            }
        )
    schema_df = pd.DataFrame(rows).sort_values(["role", "fill_rate"], ascending=[True, False])
    return {
        "n_rows": n_rows,
        "n_columns": int(df.shape[1]),
        "schema": schema_df.to_dict(orient="records"),
    }


def plot_schema_overview(schema: dict, plots_dir: Path) -> Path:
    df = pd.DataFrame(schema["schema"])
    role_counts = df["role"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    role_counts.plot(kind="barh", ax=ax, color=sns.color_palette("deep", n_colors=len(role_counts)))
    ax.set_title("Columns by role")
    ax.set_xlabel("count")
    ax.set_ylabel("")
    return _save_fig(fig, plots_dir, "01_columns_by_role")


# ── 2. Univariate distributions ───────────────────────────────────────────


def plot_continuous_distributions(df: pd.DataFrame, plots_dir: Path) -> Path:
    cols = [c for c in CONTINUOUS_PREDICTORS if c in df.columns]
    n = len(cols)
    if n == 0:
        return None
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
    axes = np.atleast_2d(axes).flatten()
    for ax, col in zip(axes, cols, strict=False):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            ax.set_title(f"{col} (empty)")
            continue
        sns.histplot(series, kde=True, ax=ax, bins=40, color="#4C72B0")
        ax.set_title(f"{col}\nn={len(series):,}  median={series.median():,.0f}")
        ax.set_xlabel("")
    for ax in axes[len(cols) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return _save_fig(fig, plots_dir, "02_continuous_distributions")


def plot_categorical_distributions(df: pd.DataFrame, plots_dir: Path) -> Path:
    cols = [
        c
        for c in (
            "country",
            "source",
            "role_family_extracted",
            "seniority_extracted",
            "remote_policy",
            "contract_type",
            "min_education",
            "manager_role",
            "posting_quality",
        )
        if c in df.columns
    ]
    n = len(cols)
    if n == 0:
        return None
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
    axes = np.atleast_2d(axes).flatten()
    for ax, col in zip(axes, cols, strict=False):
        counts = df[col].fillna("<missing>").astype(str).value_counts().head(8)
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#55A868")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes[len(cols) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return _save_fig(fig, plots_dir, "03_categorical_distributions")


# ── 3. Missingness ────────────────────────────────────────────────────────


def audit_missingness(df: pd.DataFrame) -> dict[str, Any]:
    miss_rates = (df.isna().mean() * 100).sort_values(ascending=False)
    cols_for_pattern = [c for c in ALL_PREDICTORS if c in df.columns]
    miss = df[cols_for_pattern].isna()
    # Co-missing correlations between numerical/boolean predictors only
    co_missing = miss.corr().fillna(0.0)

    # Little's MCAR test is not built into statsmodels directly. We use a
    # cheaper proxy: chi-square on missingness vs each other column to detect
    # MAR signals (missingness depends on observed values).
    mar_signals: list[dict[str, Any]] = []
    from scipy.stats import chi2_contingency  # type: ignore

    target_cols = ["country", "source", "role_family_extracted", "seniority_extracted"]
    for missing_col in (
        "min_years_experience",
        "min_education",
        TARGET_COL,
        "remote_policy",
    ):
        if missing_col not in df.columns:
            continue
        miss_indicator = df[missing_col].isna()
        for tc in target_cols:
            if tc not in df.columns:
                continue
            ct = pd.crosstab(miss_indicator, df[tc].fillna("<NA>"))
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            try:
                chi2, p, _, _ = chi2_contingency(ct)
                mar_signals.append(
                    {
                        "missing_column": missing_col,
                        "vs": tc,
                        "chi2": round(float(chi2), 2),
                        "p_value": round(float(p), 6),
                        "interpretation": (
                            "MAR-like (missingness depends on observed)"
                            if p < 0.05
                            else "no detectable dependency"
                        ),
                    }
                )
            except Exception:  # noqa: BLE001
                continue

    return {
        "missing_rates_pct": miss_rates.round(1).to_dict(),
        "co_missingness": co_missing.round(2).to_dict(),
        "mar_signals": mar_signals,
    }


def plot_missingness_matrix(df: pd.DataFrame, plots_dir: Path) -> Path:
    cols = [c for c in ALL_PREDICTORS if c in df.columns]
    sub = df[cols].head(2000)  # 2k rows is enough to see patterns
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(sub.isna(), cbar=False, ax=ax, cmap="rocket_r")
    ax.set_title("Missingness matrix (first 2,000 rows)")
    ax.set_xlabel("")
    ax.set_ylabel("row")
    plt.xticks(rotation=70, ha="right", fontsize=7)
    fig.tight_layout()
    return _save_fig(fig, plots_dir, "04_missingness_matrix")


def plot_missingness_rates(df: pd.DataFrame, plots_dir: Path) -> Path:
    cols = [c for c in ALL_PREDICTORS if c in df.columns]
    rates = df[cols].isna().mean().sort_values(ascending=True) * 100
    fig, ax = plt.subplots(figsize=(8, max(4, len(rates) * 0.25)))
    rates.plot(kind="barh", ax=ax, color="#C44E52")
    ax.set_title("Missingness rate per predictor (%)")
    ax.set_xlabel("% missing")
    fig.tight_layout()
    return _save_fig(fig, plots_dir, "05_missingness_rates")


# ── 4. Target deep-dive ───────────────────────────────────────────────────


def audit_target(df: pd.DataFrame) -> dict[str, Any]:
    y = pd.to_numeric(df.get(TARGET_COL), errors="coerce").dropna()
    if y.empty:
        return {"available": False}
    log_y = np.log10(y[y > 0])
    return {
        "available": True,
        "n": int(len(y)),
        "mean": round(float(y.mean()), 0),
        "median": round(float(y.median()), 0),
        "std": round(float(y.std()), 0),
        "p10": round(float(y.quantile(0.10)), 0),
        "p25": round(float(y.quantile(0.25)), 0),
        "p75": round(float(y.quantile(0.75)), 0),
        "p90": round(float(y.quantile(0.90)), 0),
        "p99": round(float(y.quantile(0.99)), 0),
        "skew": round(float(y.skew()), 2),
        "kurtosis": round(float(y.kurtosis()), 2),
        "log_skew": round(float(log_y.skew()), 2),
        "log_kurtosis": round(float(log_y.kurtosis()), 2),
    }


def plot_target_distribution(df: pd.DataFrame, plots_dir: Path) -> Path:
    y = pd.to_numeric(df.get(TARGET_COL), errors="coerce").dropna()
    if y.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(y, bins=60, ax=axes[0], color="#4C72B0")
    axes[0].set_title("salary_max_usd_yearly (raw, USD/year)")
    axes[0].set_xlabel("")
    axes[0].axvline(y.median(), color="red", linestyle="--", label=f"median={y.median():,.0f}")
    axes[0].legend()
    sns.histplot(np.log10(y[y > 0]), bins=60, ax=axes[1], color="#DD8452")
    axes[1].set_title("log10(salary_max_usd_yearly) — symmetric is good for ML")
    axes[1].set_xlabel("")
    fig.tight_layout()
    return _save_fig(fig, plots_dir, "06_target_distribution")


def plot_target_by_strata(df: pd.DataFrame, plots_dir: Path) -> Path:
    if TARGET_COL not in df.columns:
        return None
    sub = df.dropna(subset=[TARGET_COL]).copy()
    sub[TARGET_COL] = pd.to_numeric(sub[TARGET_COL], errors="coerce")
    sub = sub.dropna(subset=[TARGET_COL])
    if sub.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("country", axes[0, 0]),
        ("source", axes[0, 1]),
        ("role_family_extracted", axes[1, 0]),
        ("seniority_extracted", axes[1, 1]),
    ]
    for col, ax in panels:
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        order = (
            sub.groupby(col, observed=True)[TARGET_COL]
            .median()
            .sort_values(ascending=False)
            .index[:10]
        )
        sns.boxplot(
            data=sub[sub[col].isin(order)],
            x=col,
            y=TARGET_COL,
            order=order,
            ax=ax,
            showfliers=False,
        )
        ax.set_title(f"{TARGET_COL} by {col}")
        ax.set_xlabel("")
        ax.set_ylabel("USD / year")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    fig.tight_layout()
    return _save_fig(fig, plots_dir, "07_target_by_strata")


# ── 5. Bivariate ───────────────────────────────────────────────────────────


def audit_bivariate(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"continuous": [], "categorical": [], "boolean": []}
    if TARGET_COL not in df.columns:
        return out
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Continuous vs target — Pearson + Spearman correlation
    for col in CONTINUOUS_PREDICTORS:
        if col not in df.columns or col == TARGET_COL:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 30:
            continue
        out["continuous"].append(
            {
                "predictor": col,
                "n": int(mask.sum()),
                "pearson": round(float(x[mask].corr(y[mask])), 3),
                "spearman": round(float(x[mask].corr(y[mask], method="spearman")), 3),
            }
        )

    # Categorical vs target — ANOVA F-test (omnibus); report top-cardinality only
    from scipy.stats import f_oneway  # type: ignore

    for col in tuple(ORDINAL_PREDICTORS) + NOMINAL_PREDICTORS:
        if col not in df.columns:
            continue
        sub = df[[col, TARGET_COL]].dropna()
        if len(sub) < 60:
            continue
        groups = [
            pd.to_numeric(g[TARGET_COL], errors="coerce").dropna().values
            for _, g in sub.groupby(col, observed=True)
            if len(g) >= 5
        ]
        if len(groups) < 2:
            continue
        try:
            f_stat, p = f_oneway(*groups)
        except Exception:  # noqa: BLE001
            continue
        out["categorical"].append(
            {
                "predictor": col,
                "n": int(len(sub)),
                "groups": int(len(groups)),
                "anova_f": round(float(f_stat), 2),
                "p_value": round(float(p), 6),
            }
        )

    # Boolean vs target — Welch t-test
    from scipy.stats import ttest_ind  # type: ignore

    for col in BOOLEAN_PREDICTORS:
        if col not in df.columns:
            continue
        # Coerce tri-states / strings to binary indicator (any truthy = True).
        s = df[col]
        if s.dtype.name == "boolean" or pd.api.types.is_bool_dtype(s):
            yes = y[s.fillna(False).astype(bool)]
            no = y[~s.fillna(False).astype(bool)]
        else:
            yes_mask = s.fillna("").astype(str).str.lower().isin({"yes", "true", "1"})
            yes = y[yes_mask]
            no = y[~yes_mask]
        yes = yes.dropna()
        no = no.dropna()
        if len(yes) < 20 or len(no) < 20:
            continue
        try:
            t, p = ttest_ind(yes, no, equal_var=False)
        except Exception:  # noqa: BLE001
            continue
        out["boolean"].append(
            {
                "predictor": col,
                "n_yes": int(len(yes)),
                "n_no": int(len(no)),
                "median_yes": round(float(yes.median()), 0),
                "median_no": round(float(no.median()), 0),
                "t_stat": round(float(t), 2),
                "p_value": round(float(p), 6),
            }
        )

    out["continuous"].sort(key=lambda r: abs(r["spearman"]), reverse=True)
    out["categorical"].sort(key=lambda r: r["p_value"])
    out["boolean"].sort(key=lambda r: r["p_value"])
    return out


# ── 6. Multicollinearity ──────────────────────────────────────────────────


def audit_multicollinearity(df: pd.DataFrame) -> dict[str, Any]:
    # VIF dies on sparse columns once dropna() is applied across all of them
    # at once. Filter to predictors with at least 30% fill so the joint
    # complete-case sample is non-trivial.
    candidates = [c for c in CONTINUOUS_PREDICTORS if c in df.columns and c != TARGET_COL]
    candidates = [c for c in candidates if df[c].notna().mean() >= 0.30]
    if len(candidates) < 2:
        return {
            "sample": 0,
            "vif": [],
            "condition_number": None,
            "corr": {},
            "note": (
                "Fewer than 2 continuous predictors with ≥30% fill; "
                "VIF on the tabular block is uninformative until predictor coverage rises "
                "(re-run after Step 1b LLM backfill or drift in extractor coverage)."
            ),
        }
    sub = df[candidates].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.shape[0] < 30 or sub.shape[1] < 2:
        return {
            "sample": int(sub.shape[0]),
            "vif": [],
            "condition_number": None,
            "corr": {},
            "note": "joint complete-case sample too small for VIF",
        }
    corr = sub.corr().round(3)

    # VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore

    matrix = sub.values  # noqa: N806 — X would shadow nothing here, lowercase is the lint pref
    vif_rows: list[dict[str, Any]] = []
    for i, col in enumerate(sub.columns):
        try:
            vif = variance_inflation_factor(matrix, i)
        except Exception:  # noqa: BLE001
            vif = float("nan")
        vif_rows.append(
            {"predictor": col, "vif": round(float(vif), 2) if np.isfinite(vif) else None}
        )

    # Condition number of the design matrix (informational; not VIF)
    cond_num = None
    try:
        cond_num = round(float(np.linalg.cond(np.column_stack([np.ones(len(matrix)), matrix]))), 1)
    except Exception:  # noqa: BLE001
        cond_num = None

    return {
        "sample": int(sub.shape[0]),
        "vif": vif_rows,
        "condition_number": cond_num,
        "corr": corr.to_dict(),
    }


def plot_correlation_heatmap(df: pd.DataFrame, plots_dir: Path) -> Path:
    cols = [c for c in CONTINUOUS_PREDICTORS + (TARGET_COL,) if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr()
    if corr.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Correlation among continuous predictors + target")
    fig.tight_layout()
    return _save_fig(fig, plots_dir, "08_correlation_heatmap")


# ── 7. Outliers ───────────────────────────────────────────────────────────


def audit_outliers(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in CONTINUOUS_PREDICTORS:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if x.empty:
            continue
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = (x - x.mean()) / x.std() if x.std() > 0 else x * 0
        out[col] = {
            "n": int(len(x)),
            "iqr_low_fence": round(float(low_fence), 1),
            "iqr_high_fence": round(float(high_fence), 1),
            "n_iqr_outliers": int(((x < low_fence) | (x > high_fence)).sum()),
            "n_z_above_3": int((z.abs() > 3).sum()),
            "min": round(float(x.min()), 1),
            "max": round(float(x.max()), 1),
        }
    return out


def plot_target_qq(df: pd.DataFrame, plots_dir: Path) -> Path:
    """Q-Q plot for the target on raw + log scales.

    We deliberately skip Shapiro-Wilk / Anderson-Darling — at our n (~6k
    disclosed rows) those tests reject normality for any tiny deviation,
    so they're uninformative as a transformation decision. The visual
    Q-Q + skew/kurtosis effect sizes (already in metrics.json) settle the
    log-transform choice cleanly.
    """
    from scipy import stats  # type: ignore

    y = pd.to_numeric(df.get(TARGET_COL), errors="coerce").dropna()
    if y.empty:
        return None
    y_pos = y[y > 0]
    log_y = np.log10(y_pos)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    stats.probplot(y_pos, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q raw — strong departure in the upper tail")
    axes[0].get_lines()[0].set_color("#4C72B0")
    axes[0].get_lines()[0].set_markersize(3)
    stats.probplot(log_y, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q log10 — much closer to the reference line")
    axes[1].get_lines()[0].set_color("#DD8452")
    axes[1].get_lines()[0].set_markersize(3)
    fig.tight_layout()
    return _save_fig(fig, plots_dir, "10_target_qq")


def plot_pca_continuous(df: pd.DataFrame, plots_dir: Path) -> Path:
    """PCA on the well-populated continuous block, projected to 2-D and
    colored by target. Closes the multivariate gap flagged in §15.3 of
    the literature review.

    Caveat: only 2-3 well-populated continuous predictors exist post-EDA
    (min_years_experience and salary_min_usd_yearly), so the 2-D PCA
    projection mostly recovers the original axes. The plot is still
    useful as a sanity check before adding the 1024-dim bge-m3 embedding
    in Phase 5 (where PCA is unambiguously load-bearing).
    """
    from sklearn.decomposition import PCA  # type: ignore

    cols = [
        c
        for c in CONTINUOUS_PREDICTORS
        if c in df.columns and c != TARGET_COL and df[c].notna().mean() >= 0.30
    ]
    if len(cols) < 2 or TARGET_COL not in df.columns:
        return None
    sub = df[cols + [TARGET_COL]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 30:
        return None

    matrix = sub[cols].values
    matrix_std = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(matrix_std)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=np.log10(sub[TARGET_COL].values),
        cmap="viridis",
        s=10,
        alpha=0.6,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(salary_max_usd_yearly)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title(f"PCA of standardized continuous predictors\n({', '.join(cols)})")
    fig.tight_layout()
    return _save_fig(fig, plots_dir, "11_pca_continuous")


def plot_target_outliers(df: pd.DataFrame, plots_dir: Path) -> Path:
    y = pd.to_numeric(df.get(TARGET_COL), errors="coerce").dropna()
    if y.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(x=y, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"{TARGET_COL} — boxplot (outliers shown)")
    axes[0].set_xlabel("USD / year")
    sns.boxplot(x=np.log10(y[y > 0]), ax=axes[1], color="#DD8452")
    axes[1].set_title("log10(target) — boxplot")
    axes[1].set_xlabel("")
    fig.tight_layout()
    return _save_fig(fig, plots_dir, "09_target_outliers")


# ── Orchestrator ───────────────────────────────────────────────────────────


def run_audit(input_path: Path, output_dir: Path) -> dict[str, Any]:
    _set_style()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("loaded %d rows, %d columns", *df.shape)

    schema = audit_schema(df)
    plot_schema_overview(schema, plots_dir)
    plot_continuous_distributions(df, plots_dir)
    plot_categorical_distributions(df, plots_dir)

    missing = audit_missingness(df)
    plot_missingness_matrix(df, plots_dir)
    plot_missingness_rates(df, plots_dir)

    target = audit_target(df)
    plot_target_distribution(df, plots_dir)
    plot_target_by_strata(df, plots_dir)

    bivariate = audit_bivariate(df)
    multicollinearity = audit_multicollinearity(df)
    plot_correlation_heatmap(df, plots_dir)

    outliers = audit_outliers(df)
    plot_target_outliers(df, plots_dir)
    plot_target_qq(df, plots_dir)
    plot_pca_continuous(df, plots_dir)

    metrics = {
        "schema": schema,
        "missingness": missing,
        "target": target,
        "bivariate": bivariate,
        "multicollinearity": multicollinearity,
        "outliers": outliers,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    logger.info("wrote %s", metrics_path)

    report_path = output_dir / "report.md"
    write_report(metrics, plots_dir, report_path)
    logger.info("wrote %s", report_path)

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/curated/jobs.parquet")
    p.add_argument("--output-dir", default="data/eda")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    run_audit(Path(args.input), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
