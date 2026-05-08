"""Feature encoding pipelines for the salary regressor.

Two encoders, both fit-on-train-only:

- ``fit_mincer_encoder`` — narrow Mincer-equation feature set
  (years_experience + education + country) for Tier 2.
- ``fit_full_encoder`` — full ColumnTransformer applying the recommendations
  in ``LITERATURE_REVIEW.md`` §14 (one-hot / ordinal-int / k-fold target
  encoding / multi-hot lists / cyclic + days-since dates) for Tiers 3-5.

Lists are multi-hotted to a top-N vocabulary inferred from training data.
Datetime is decomposed into ``posted_month`` (cyclic sin/cos) + ``days_since_posted``.
Booleans are 0/1 with a separate ``_isna`` indicator column for non-tree models.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

logger = logging.getLogger("models.salary.encode")

# Ordinal level orderings (matches eda/audit.py and LITERATURE_REVIEW.md §3).
ORDINAL_LEVELS: dict[str, list[str]] = {
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

LOW_CARD_NOMINAL: tuple[str, ...] = (
    "country",
    "source",
    "role_family_extracted",
    "remote_policy",
    "contract_type",
    "equity_form",
    "bonus_type",
)
HIGH_CARD_NOMINAL: tuple[str, ...] = ("region", "city")
CONTINUOUS_COLS: tuple[str, ...] = ("min_years_experience",)
BOOLEAN_COLS: tuple[str, ...] = (
    "requires_security_clearance",
    "offers_relocation",
    "offers_equity",
    "bonus_mentioned",
    "on_call_required",
)
TRI_STATE_COLS: tuple[str, ...] = ("offers_visa_sponsorship",)

# Multi-hot lists: top-N tokens by training-set frequency.
TECH_STACK_TOP_N: int = 25
LANG_TOP_N: int = 5
CITIZENSHIP_TOP_N: int = 4


# ── Custom transformers ────────────────────────────────────────────────────


class ListMultiHotEncoder(BaseEstimator, TransformerMixin):
    """Multi-hot encode a list-valued column to a fixed top-N vocabulary."""

    def __init__(self, top_n: int = 25, prefix: str = "list") -> None:
        self.top_n = top_n
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y=None) -> ListMultiHotEncoder:
        assert X.shape[1] == 1, "expected single column"
        col = X.iloc[:, 0]
        counts = Counter()
        for items in col.dropna():
            if isinstance(items, (list, tuple, np.ndarray)):
                counts.update(str(x) for x in items if x)
        self.vocab_ = [t for t, _ in counts.most_common(self.top_n)]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        col = X.iloc[:, 0]
        out = np.zeros((len(col), len(self.vocab_) + 1), dtype=np.int8)
        idx = {t: i for i, t in enumerate(self.vocab_)}
        for row, items in enumerate(col):
            if not isinstance(items, (list, tuple, np.ndarray)):
                continue
            present = False
            for item in items:
                key = str(item)
                if key in idx:
                    out[row, idx[key]] = 1
                    present = True
            out[row, -1] = int(present)  # 'has_any' indicator
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = [f"{self.prefix}__{t}" for t in self.vocab_]
        names.append(f"{self.prefix}__has_any")
        return np.asarray(names, dtype=object)


class TechStackEncoder(BaseEstimator, TransformerMixin):
    """Multi-hot top-N tech stack tokens + count + has_modern_ml flag."""

    MODERN_ML_TOKENS: frozenset[str] = frozenset(
        {
            "PyTorch",
            "TensorFlow",
            "HuggingFace",
            "MLflow",
            "Weights & Biases",
            "Spark",
            "Databricks",
            "LLMs",
            "RAG",
            "scikit-learn",
            "XGBoost",
        }
    )

    def __init__(self, top_n: int = 25) -> None:
        self.top_n = top_n
        self._inner = ListMultiHotEncoder(top_n=top_n, prefix="tech")

    def fit(self, X: pd.DataFrame, y=None) -> TechStackEncoder:
        self._inner.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        multi_hot = self._inner.transform(X)
        col = X.iloc[:, 0]
        counts = np.zeros((len(col), 1), dtype=np.int16)
        modern_flag = np.zeros((len(col), 1), dtype=np.int8)
        for row, items in enumerate(col):
            if not isinstance(items, (list, tuple, np.ndarray)):
                continue
            tokens = [str(x) for x in items if x]
            counts[row, 0] = len(tokens)
            if any(t in self.MODERN_ML_TOKENS for t in tokens):
                modern_flag[row, 0] = 1
        return np.hstack([multi_hot, counts, modern_flag])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(self._inner.get_feature_names_out())
        names.extend(["tech__count", "tech__has_modern_ml"])
        return np.asarray(names, dtype=object)


class DatetimeFeaturizer(BaseEstimator, TransformerMixin):
    """Decompose `posted_at` into cyclic month + days_since_posted."""

    def __init__(self, reference_date: pd.Timestamp | None = None) -> None:
        self.reference_date = reference_date

    def fit(self, X: pd.DataFrame, y=None) -> DatetimeFeaturizer:
        col = pd.to_datetime(X.iloc[:, 0], utc=True, errors="coerce")
        self.reference_date_ = self.reference_date if self.reference_date is not None else col.max()
        if pd.isna(self.reference_date_):
            self.reference_date_ = pd.Timestamp.now(tz="UTC")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        col = pd.to_datetime(X.iloc[:, 0], utc=True, errors="coerce")
        days = (self.reference_date_ - col).dt.days.astype("Float64")
        days = days.fillna(days.median()).astype(float).clip(lower=0).values
        month = col.dt.month.fillna(6).astype(int).values
        sin_m = np.sin(2 * np.pi * month / 12)
        cos_m = np.cos(2 * np.pi * month / 12)
        is_missing = col.isna().astype(np.int8).values
        return np.column_stack([days, sin_m, cos_m, is_missing])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(
            ["posted__days_since", "posted__sin_month", "posted__cos_month", "posted__is_missing"],
            dtype=object,
        )


class TriStateOneHot(BaseEstimator, TransformerMixin):
    """One-hot a tri-state field (yes / no / unspecified / missing)."""

    LEVELS: tuple[str, ...] = ("yes", "no", "unspecified")

    def fit(self, X: pd.DataFrame, y=None) -> TriStateOneHot:
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        col = X.iloc[:, 0].fillna("missing").astype(str)
        out = np.zeros((len(col), len(self.LEVELS) + 1), dtype=np.int8)
        for i, level in enumerate(self.LEVELS):
            out[:, i] = (col == level).astype(np.int8)
        out[:, -1] = (~col.isin(self.LEVELS)).astype(np.int8)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(
            [f"sponsorship__{level}" for level in self.LEVELS] + ["sponsorship__missing"],
            dtype=object,
        )


# ── Encoder factories ──────────────────────────────────────────────────────


@dataclass(slots=True)
class FittedEncoder:
    """Holds a fitted ColumnTransformer + feature-name accessor."""

    transformer: ColumnTransformer
    feature_names: list[str]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.transformer.transform(X)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        return pd.DataFrame(arr, columns=self.feature_names, index=X.index)


def _build_full_transformer(X_train: pd.DataFrame) -> ColumnTransformer:
    """Assemble the full ColumnTransformer for Tier 3-5 models."""
    transformers: list[tuple[str, Any, list[str] | str]] = []

    # Continuous: median-impute, then scale (helps Ridge; tree models ignore
    # both, but cost is negligible).
    cont_cols = [c for c in CONTINUOUS_COLS if c in X_train.columns]
    if cont_cols:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                cont_cols,
            ),
        )

    # Ordinals: integer-encode in known order; unknown levels → -1; NaN → -1
    for col, levels in ORDINAL_LEVELS.items():
        if col in X_train.columns:
            transformers.append(
                (
                    f"ord_{col}",
                    OrdinalEncoder(
                        categories=[levels],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                    [col],
                ),
            )

    # Low-card nominal: one-hot
    low_card = [c for c in LOW_CARD_NOMINAL if c in X_train.columns]
    if low_card:
        transformers.append(
            (
                "low_card_ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
                low_card,
            ),
        )

    # High-card nominal: k-fold target encoding (sklearn 1.3+ does it natively)
    high_card = [c for c in HIGH_CARD_NOMINAL if c in X_train.columns]
    if high_card:
        transformers.append(
            (
                "high_card_target",
                TargetEncoder(target_type="continuous", smooth="auto", random_state=42),
                high_card,
            ),
        )

    # Booleans → 0/1
    bool_cols = [c for c in BOOLEAN_COLS if c in X_train.columns]
    if bool_cols:
        transformers.append(("booleans", _BooleanCleaner(), bool_cols))

    # Tri-state sponsorship
    if "offers_visa_sponsorship" in X_train.columns:
        transformers.append(
            ("tri_state", TriStateOneHot(), ["offers_visa_sponsorship"]),
        )

    # Lists
    if "tech_stack" in X_train.columns:
        transformers.append(
            ("tech_stack", TechStackEncoder(top_n=TECH_STACK_TOP_N), ["tech_stack"])
        )
    if "language_requirements" in X_train.columns:
        transformers.append(
            (
                "languages",
                ListMultiHotEncoder(top_n=LANG_TOP_N, prefix="lang"),
                ["language_requirements"],
            ),
        )
    if "requires_citizenship" in X_train.columns:
        transformers.append(
            (
                "citizenship",
                ListMultiHotEncoder(top_n=CITIZENSHIP_TOP_N, prefix="citizenship"),
                ["requires_citizenship"],
            ),
        )

    # Datetime
    if "posted_at" in X_train.columns:
        transformers.append(("posted_at", DatetimeFeaturizer(), ["posted_at"]))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


class _BooleanCleaner(BaseEstimator, TransformerMixin):
    """Coerce nullable-boolean / object columns to 0/1 + per-column isna mask."""

    def fit(self, X: pd.DataFrame, y=None) -> _BooleanCleaner:
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out_cols: list[np.ndarray] = []
        for col in self.columns_:
            s = X[col]
            is_na = s.isna().to_numpy().astype(np.int8)
            filled = (
                s.fillna(False)
                .map({True: 1, False: 0, "true": 1, "false": 0})
                .fillna(0)
                .astype(np.int8)
                .to_numpy()
            )
            out_cols.append(filled)
            out_cols.append(is_na)
        return np.column_stack(out_cols)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names: list[str] = []
        for col in self.columns_:
            names.append(col)
            names.append(f"{col}__isna")
        return np.asarray(names, dtype=object)


def fit_full_encoder(X_train: pd.DataFrame, y_train: pd.Series) -> FittedEncoder:
    """Fit the full encoder on training data only."""
    transformer = _build_full_transformer(X_train)
    transformer.fit(X_train, y_train)
    feature_names = list(transformer.get_feature_names_out())
    logger.info(
        "fit_full_encoder :: %d input columns → %d encoded features",
        X_train.shape[1],
        len(feature_names),
    )
    return FittedEncoder(transformer=transformer, feature_names=feature_names)


def fit_mincer_encoder(X_train: pd.DataFrame, y_train: pd.Series) -> FittedEncoder:
    """Fit the narrow Mincer feature set: years_experience + experience² +
    education (ordinal int) + country (one-hot)."""
    cols_present = [
        c for c in ("min_years_experience", "min_education", "country") if c in X_train.columns
    ]
    if "min_years_experience" not in cols_present:
        raise ValueError("min_years_experience missing; cannot fit Mincer encoder")

    transformers = []
    transformers.append(("yoe", _MincerExperience(), ["min_years_experience"]))
    if "min_education" in cols_present:
        transformers.append(
            (
                "edu",
                OrdinalEncoder(
                    categories=[ORDINAL_LEVELS["min_education"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                ["min_education"],
            )
        )
    if "country" in cols_present:
        transformers.append(
            ("country", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["country"])
        )
    transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    transformer.fit(X_train, y_train)
    feature_names = list(transformer.get_feature_names_out())
    logger.info("fit_mincer_encoder :: %d Mincer features", len(feature_names))
    return FittedEncoder(transformer=transformer, feature_names=feature_names)


class _MincerExperience(BaseEstimator, TransformerMixin):
    """Years of experience + experience² + missingness indicator (the
    classic Mincer 1974 functional form)."""

    def fit(self, X: pd.DataFrame, y=None) -> _MincerExperience:
        col = pd.to_numeric(X.iloc[:, 0], errors="coerce")
        self.median_ = float(col.median()) if col.notna().any() else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        col = pd.to_numeric(X.iloc[:, 0], errors="coerce")
        is_na = col.isna().astype(np.int8).values
        filled = col.fillna(self.median_).astype(float).values
        return np.column_stack([filled, filled**2, is_na])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(["yoe", "yoe_sq", "yoe_isna"], dtype=object)


def stratify_label(strata: Iterable[str]) -> pd.Series:
    """Helper for downstream eval: convert ``country/source`` to a Series."""
    return pd.Series(list(strata), name="stratum")
