"""Build the modelling dataset from the curated layer.

Reads ``data/curated/jobs.parquet`` and ``data/eda/test_split_ids.json``,
filters to disclosed-salary rows whose ``posting_quality == 'real'``, applies
sanity filters (winsorize target at the 99.5th percentile, drop rows with
annualized salary < $30k), and emits ``(X_train, X_test, y_train, y_test,
strata_train, strata_test)`` as plain DataFrames / Series.

The frozen test split from ``eda.split.freeze_split`` is the **only** train /
test boundary used downstream — the audit step (Step 2.5) committed the
manifest specifically to lock this contract.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("models.salary.dataset")

DEFAULT_CURATED_PATH = Path("data/curated/jobs.parquet")
DEFAULT_SPLIT_PATH = Path("data/eda/test_split_ids.json")
TARGET_COL = "salary_max_usd_yearly"
LOG_TARGET_COL = "log10_salary_max_usd_yearly"

# Per LITERATURE_REVIEW.md §12: winsorize at the 99.5th percentile
# of the disclosed-salary distribution to clip clear typos / total-comp leaks
# without losing the senior-IC right tail.
WINSORIZE_QUANTILE = 0.995

# Floor: anything below ~$30k/yr USD-equivalent is likely a hourly-period
# misclassification or part-time posting we can't model meaningfully.
MIN_PLAUSIBLE_USD = 30_000.0


@dataclass(slots=True, frozen=True)
class SalaryDataset:
    """Materialized train / test split for the salary regressor."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series  # log10 USD-yearly
    y_test: pd.Series  # log10 USD-yearly
    y_train_raw: pd.Series  # USD/year, back-transform reference
    y_test_raw: pd.Series
    strata_train: pd.Series  # "country/source" string for stratified eval
    strata_test: pd.Series
    feature_cols: list[str]
    info: dict[str, Any]


def load_test_ids(split_path: Path = DEFAULT_SPLIT_PATH) -> set[str]:
    if not split_path.exists():
        raise FileNotFoundError(
            f"frozen split manifest missing at {split_path} — "
            "run `uv run python -m eda.split` first"
        )
    return set(json.loads(split_path.read_text())["test_ids"])


def _filter_for_modelling(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the rules from `LITERATURE_REVIEW.md` §12 + the
    `posting_quality == 'real'` row filter from §6 of the EDA report.
    Returns (clean_df, drop_counts)."""
    drops: dict[str, int] = {"input": len(df)}

    df = df[df["posting_quality"] == "real"]
    drops["after_real_only"] = len(df)

    df = df[df["salary_disclosed"].fillna(False).astype(bool)]
    drops["after_disclosed"] = len(df)

    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] >= MIN_PLAUSIBLE_USD]
    drops["after_min_plausible"] = len(df)

    cap = df[TARGET_COL].quantile(WINSORIZE_QUANTILE)
    df = df.copy()
    df.loc[df[TARGET_COL] > cap, TARGET_COL] = cap
    drops["winsorize_cap_usd"] = int(round(cap))
    drops["final_rows"] = len(df)
    return df, drops


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pull the predictor columns from the curated table.

    Returns a DataFrame whose columns are all candidate predictors per
    `DATA_DICTIONARY.md` §10 ("Inclusion summary") — ordinal/nominal still as
    raw strings; encoders apply in ``models.salary.encode``.
    """
    feature_cols = [
        # Continuous
        "min_years_experience",
        # Ordinal
        "min_education",
        "seniority_extracted",
        "manager_role",
        "clearance_level",
        # Low-card nominal
        "country",
        "source",
        "role_family_extracted",
        "remote_policy",
        "contract_type",
        "equity_form",
        "bonus_type",
        # High-card nominal — encoded via target encoding inside the encoder
        "region",
        "city",
        # Boolean / tri-state
        "requires_security_clearance",
        "offers_visa_sponsorship",
        "offers_relocation",
        "offers_equity",
        "bonus_mentioned",
        "on_call_required",
        # Lists
        "requires_citizenship",
        "language_requirements",
        "tech_stack",
        # Datetime
        "posted_at",
    ]
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("missing predictor columns: %s", missing)
    return df[available].copy()


def build_dataset(
    curated_path: Path = DEFAULT_CURATED_PATH,
    split_path: Path = DEFAULT_SPLIT_PATH,
) -> SalaryDataset:
    df = pd.read_parquet(curated_path)
    logger.info("loaded curated %d rows from %s", len(df), curated_path)
    test_ids = load_test_ids(split_path)
    logger.info("frozen test set: %d ids", len(test_ids))

    df, drops = _filter_for_modelling(df)
    logger.info("after modelling filters :: %s", drops)

    X = _build_features(df)
    y_raw = df[TARGET_COL].astype(float)
    y_log = np.log10(y_raw)
    strata = df["country"].astype(str) + "/" + df["source"].astype(str)

    is_test = df["id"].astype(str).isin(test_ids)

    info = {
        "drops": drops,
        "n_train": int((~is_test).sum()),
        "n_test": int(is_test.sum()),
        "test_frac_actual": round(float(is_test.mean()), 4),
        "feature_cols": list(X.columns),
        "target_col": TARGET_COL,
        "log_target": True,
        "winsorize_quantile": WINSORIZE_QUANTILE,
        "min_plausible_usd": MIN_PLAUSIBLE_USD,
    }
    logger.info(
        "dataset materialised :: train=%d, test=%d (%.1f%% holdout)",
        info["n_train"],
        info["n_test"],
        info["test_frac_actual"] * 100,
    )

    return SalaryDataset(
        X_train=X[~is_test].reset_index(drop=True),
        X_test=X[is_test].reset_index(drop=True),
        y_train=y_log[~is_test].reset_index(drop=True),
        y_test=y_log[is_test].reset_index(drop=True),
        y_train_raw=y_raw[~is_test].reset_index(drop=True),
        y_test_raw=y_raw[is_test].reset_index(drop=True),
        strata_train=strata[~is_test].reset_index(drop=True),
        strata_test=strata[is_test].reset_index(drop=True),
        feature_cols=list(X.columns),
        info=info,
    )
