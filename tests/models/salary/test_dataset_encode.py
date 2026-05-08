"""Tests for ``models.salary.dataset`` and ``models.salary.encode``.

Synthetic data only — we don't read the real curated parquet, so these tests
run independently of any locally-cached data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.salary.encode import (
    DatetimeFeaturizer,
    ListMultiHotEncoder,
    TechStackEncoder,
    TriStateOneHot,
    fit_full_encoder,
    fit_mincer_encoder,
)


def _make_synthetic(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "min_years_experience": rng.integers(1, 15, n).astype(float),
            "min_education": rng.choice(
                ["bachelors", "masters", "phd", None],
                p=[0.5, 0.3, 0.1, 0.1],
                size=n,
            ),
            "seniority_extracted": rng.choice(
                ["junior", "mid", "senior", "staff"],
                size=n,
            ),
            "manager_role": rng.choice(["ic", "manager", None], p=[0.7, 0.2, 0.1], size=n),
            "clearance_level": rng.choice([None, "secret", "ts_sci"], p=[0.9, 0.07, 0.03], size=n),
            "country": rng.choice(["US", "CA"], p=[0.95, 0.05], size=n),
            "source": rng.choice(["greenhouse", "lever", "ashby"], size=n),
            "role_family_extracted": rng.choice(
                ["DS", "MLE", "DE", "Other", "Manager"],
                size=n,
            ),
            "remote_policy": rng.choice([None, "remote-na", "hybrid", "onsite"], size=n),
            "contract_type": rng.choice([None, "full_time", "internship"], size=n),
            "equity_form": rng.choice([None, "rsu", "options"], size=n),
            "bonus_type": rng.choice([None, "annual", "signing"], size=n),
            "region": rng.choice(["CA", "NY", "WA", "ON", "BC", None], size=n),
            "city": rng.choice(["San Francisco", "New York", "Toronto", None], size=n),
            "requires_security_clearance": rng.choice([True, False, None], size=n),
            "offers_visa_sponsorship": rng.choice(["yes", "no", "unspecified", None], size=n),
            "offers_relocation": rng.choice([True, False, None], size=n),
            "offers_equity": rng.choice([True, False, None], size=n),
            "bonus_mentioned": rng.choice([True, False, None], size=n),
            "on_call_required": rng.choice([True, False, None], size=n),
            "requires_citizenship": [["US"] if x < 0.1 else None for x in rng.random(n)],
            "language_requirements": [["en"] if x < 0.5 else None for x in rng.random(n)],
            "tech_stack": [
                ["Python", "AWS"] if x < 0.5 else ["Python"] if x < 0.8 else None
                for x in rng.random(n)
            ],
            "posted_at": pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
        }
    )
    y = pd.Series(np.log10(rng.uniform(80_000, 300_000, n)))
    return df, y


def test_mincer_encoder_produces_six_features():
    X, y = _make_synthetic()
    enc = fit_mincer_encoder(X, y)
    Xt = enc.transform(X)
    assert Xt.shape[1] == len(enc.feature_names) == 6  # yoe, yoe², isna, edu, country×2
    assert "yoe_sq" in enc.feature_names
    assert any(name.startswith("country") for name in enc.feature_names)


def test_full_encoder_handles_lists_and_dates_and_nans():
    X, y = _make_synthetic()
    enc = fit_full_encoder(X, y)
    Xt = enc.transform(X)
    # Must not have NaNs after encoding (Ridge would reject them).
    assert not Xt.isna().any().any()
    assert Xt.shape[0] == len(X)
    # Tech stack expanded to ≥ 1 columns + count + has_modern_ml
    tech_cols = [c for c in Xt.columns if c.startswith("tech")]
    assert "tech__count" in tech_cols
    assert "tech__has_modern_ml" in tech_cols


def test_full_encoder_includes_all_expected_blocks():
    X, y = _make_synthetic()
    enc = fit_full_encoder(X, y)
    names = enc.feature_names
    # Continuous (scaled), ordinal, one-hot, target, booleans, lists, datetime
    assert "min_years_experience" in names
    assert any(n.startswith("country_") for n in names)
    assert any(n.startswith("posted__") for n in names)
    # Tri-state sponsorship
    assert any(n.startswith("sponsorship__") for n in names)


def test_list_multihot_encoder_top_n():
    enc = ListMultiHotEncoder(top_n=3, prefix="t").fit(
        pd.DataFrame({"x": [["a", "b"], ["a"], ["c"], ["a", "c"], ["d", "e", "f"]]})
    )
    out = enc.transform(pd.DataFrame({"x": [["a", "z"], None]}))
    # Vocab is {a, c} or {a, c, b} — must not include 'z' or None
    feat = list(enc.get_feature_names_out())
    assert all(f.startswith("t__") for f in feat)
    # Last column is "has_any"
    assert feat[-1].endswith("has_any")
    assert out[0, -1] == 1  # row 0 had at least one in-vocab token
    assert out[1, -1] == 0  # row 1 was None


def test_tech_stack_encoder_modern_ml_flag():
    X = pd.DataFrame({"tech": [["PyTorch", "Python"], ["Excel"], None]})
    enc = TechStackEncoder(top_n=10).fit(X)
    out = enc.transform(X)
    feat = enc.get_feature_names_out()
    has_modern_idx = list(feat).index("tech__has_modern_ml")
    count_idx = list(feat).index("tech__count")
    assert out[0, has_modern_idx] == 1  # PyTorch is in MODERN_ML_TOKENS
    assert out[1, has_modern_idx] == 0
    assert out[0, count_idx] == 2
    assert out[1, count_idx] == 1


def test_datetime_featurizer_decomposes():
    df = pd.DataFrame({"d": pd.to_datetime(["2026-01-15", "2026-07-15", None], utc=True)})
    enc = DatetimeFeaturizer().fit(df)
    out = enc.transform(df)
    feat = list(enc.get_feature_names_out())
    assert feat == [
        "posted__days_since",
        "posted__sin_month",
        "posted__cos_month",
        "posted__is_missing",
    ]
    assert out.shape == (3, 4)
    assert out[2, 3] == 1  # is_missing for the None row


def test_tri_state_one_hot_handles_missing():
    X = pd.DataFrame({"x": ["yes", "no", "unspecified", None, "weird"]})
    enc = TriStateOneHot().fit(X)
    out = enc.transform(X)
    feat = list(enc.get_feature_names_out())
    assert feat == [
        "sponsorship__yes",
        "sponsorship__no",
        "sponsorship__unspecified",
        "sponsorship__missing",
    ]
    # Row 3 has NaN → missing. Row 4 has "weird" → also missing (unknown level).
    assert out[3, 3] == 1
    assert out[4, 3] == 1


def test_mincer_requires_yoe_column():
    X = pd.DataFrame({"min_education": ["bachelors"]})
    y = pd.Series([5.0])
    with pytest.raises(ValueError, match="min_years_experience"):
        fit_mincer_encoder(X, y)
