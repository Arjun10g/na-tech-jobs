"""Smoke + correctness tests for baselines and eval helpers.

We deliberately keep the model fixtures small and synthetic so CI runs
don't require sklearn/xgboost (the dev group doesn't pull the [ml] extras).
The full-ladder integration runs locally / in a separate workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.salary.baselines import ConstantBaseline, StratifiedMeanBaseline
from models.salary.eval import (
    bootstrap_mae_ci,
    evaluate_tier,
    leaderboard,
    mae_usd,
    mape_pct,
    r2_log,
)

# ── baselines ─────────────────────────────────────────────────────────────


def test_constant_baseline_predicts_train_mean():
    X = pd.DataFrame({"x": range(100)})
    y = pd.Series(np.linspace(5.0, 5.4, 100))
    m = ConstantBaseline().fit(X, y)
    preds = m.predict(X.head(10))
    assert preds.shape == (10,)
    assert np.allclose(preds, y.mean())


def test_stratified_mean_uses_per_stratum_mean():
    n = 300
    rng = np.random.default_rng(0)
    strata = pd.Series(rng.choice(["a", "b"], size=n))
    y = pd.Series([5.5 if s == "a" else 5.1 for s in strata]) + rng.normal(0, 0.01, n)
    X = pd.DataFrame({"x": range(n)})
    m = StratifiedMeanBaseline(smoothing=5).fit(X, y, strata=strata)
    preds = m.predict(X, strata=strata)
    assert np.allclose(preds[strata == "a"].mean(), 5.5, atol=0.05)
    assert np.allclose(preds[strata == "b"].mean(), 5.1, atol=0.05)


def test_stratified_mean_falls_back_for_unseen_stratum():
    train_strata = pd.Series(["a"] * 50)
    y_train = pd.Series([5.3] * 50)
    test_strata = pd.Series(["unseen"] * 5)
    X = pd.DataFrame({"x": range(50)})
    Xt = pd.DataFrame({"x": range(5)})
    m = StratifiedMeanBaseline().fit(X, y_train, strata=train_strata)
    preds = m.predict(Xt, strata=test_strata)
    assert np.allclose(preds, m.global_mean_)


# ── eval metrics ──────────────────────────────────────────────────────────


def test_mae_usd_perfect_predictions_zero():
    y_log = np.log10(np.array([100_000, 200_000, 150_000]))
    assert mae_usd(y_log, y_log) == 0.0


def test_mae_usd_back_transformation():
    y_log = np.log10(np.array([100_000.0, 200_000.0]))
    yp_log = np.log10(np.array([110_000.0, 220_000.0]))
    assert mae_usd(y_log, yp_log) == pytest.approx((10_000 + 20_000) / 2)


def test_mape_pct_units():
    y_log = np.log10(np.array([100_000.0]))
    yp_log = np.log10(np.array([110_000.0]))
    assert mape_pct(y_log, yp_log) == pytest.approx(10.0)


def test_r2_log_perfect_one():
    y_log = np.log10(np.array([100_000.0, 200_000.0, 150_000.0]))
    assert r2_log(y_log, y_log) == pytest.approx(1.0)


def test_r2_log_constant_pred_zero():
    y_log = np.array([5.0, 5.5, 6.0])
    pred = np.full_like(y_log, y_log.mean())
    assert r2_log(y_log, pred) == pytest.approx(0.0)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(42)
    n = 200
    y_log = rng.normal(5.3, 0.1, n)
    pred = y_log + rng.normal(0, 0.05, n)
    ci = bootstrap_mae_ci(y_log, pred, n_bootstrap=200, seed=0)
    assert ci["mae_ci_low"] <= ci["mae"] <= ci["mae_ci_high"]
    assert ci["n"] == n


def test_evaluate_tier_returns_stratified_breakdown():
    rng = np.random.default_rng(0)
    n = 300
    y_log = rng.normal(5.3, 0.1, n)
    pred = y_log + rng.normal(0, 0.05, n)
    strata = pd.Series(rng.choice(["US/greenhouse", "CA/ashby"], size=n))
    res = evaluate_tier("test", pd.Series(y_log), pred, strata, n_bootstrap=100)
    assert res.n_test == n
    assert "stratum" in res.stratified.columns
    assert set(res.stratified["stratum"]) <= {"US/greenhouse", "CA/ashby"}


def test_leaderboard_sorts_by_mae_ascending():
    res_a = evaluate_tier(
        "a",
        pd.Series([5.0, 5.1]),
        np.array([5.0, 5.0]),
        pd.Series(["x", "x"]),
        n_bootstrap=10,
    )
    res_b = evaluate_tier(
        "b",
        pd.Series([5.0, 5.1]),
        np.array([5.0, 5.1]),
        pd.Series(["x", "x"]),
        n_bootstrap=10,
    )
    lb = leaderboard([res_a, res_b])
    assert lb.iloc[0]["tier"] == "b"  # tier b has zero MAE
