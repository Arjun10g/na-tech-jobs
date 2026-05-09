"""Tests for monitoring.champion_challenger.

Locks the promotion rule from CLAUDE.md §7:

  Promote only if primary lifts ≥1% AND no secondary regresses >2%.
"""

from __future__ import annotations

import pytest

from monitoring.champion_challenger import (
    DEFAULT_PRIMARY_LIFT_THRESHOLD,
    DEFAULT_SECONDARY_REGRESSION_TOLERANCE,
    gate,
    gate_classifier,
    gate_salary_regressor,
)


def _summary(eval_metrics: dict) -> dict:
    return {"eval": eval_metrics}


# ── Higher-is-better (classifier-style) ───────────────────────────────────


def test_promotes_when_primary_lifts_above_threshold():
    champ = _summary({"eval_f1_macro": 0.80, "eval_accuracy": 0.85, "eval_f1_weighted": 0.82})
    chal = _summary({"eval_f1_macro": 0.83, "eval_accuracy": 0.86, "eval_f1_weighted": 0.83})
    d = gate_classifier("seniority", champ, chal)
    assert d.promote is True
    assert "0.80" not in d.reason  # reason references the lift, not raw value
    assert d.primary.delta == pytest.approx(0.03, abs=1e-6)
    assert d.primary.delta_pct == pytest.approx(0.0375, abs=1e-3)


def test_holds_when_primary_lift_below_threshold():
    champ = _summary({"eval_f1_macro": 0.80, "eval_accuracy": 0.85, "eval_f1_weighted": 0.82})
    chal = _summary(
        {"eval_f1_macro": 0.804, "eval_accuracy": 0.86, "eval_f1_weighted": 0.83}
    )  # +0.5%
    d = gate_classifier("seniority", champ, chal)
    assert d.promote is False
    assert "primary" in d.reason


def test_holds_when_primary_drops():
    champ = _summary({"eval_f1_macro": 0.80, "eval_accuracy": 0.85, "eval_f1_weighted": 0.82})
    chal = _summary(
        {"eval_f1_macro": 0.78, "eval_accuracy": 0.86, "eval_f1_weighted": 0.83}
    )  # -2.5%
    d = gate_classifier("seniority", champ, chal)
    assert d.promote is False
    assert d.primary.delta < 0


def test_holds_on_secondary_regression_above_tolerance():
    champ = _summary({"eval_f1_macro": 0.80, "eval_accuracy": 0.90, "eval_f1_weighted": 0.85})
    chal = _summary(
        {"eval_f1_macro": 0.84, "eval_accuracy": 0.85, "eval_f1_weighted": 0.85}
    )  # acc -5.6%
    d = gate_classifier("seniority", champ, chal)
    assert d.promote is False
    assert "secondary" in d.reason
    assert "eval_accuracy" in d.reason


def test_promotes_when_secondary_regresses_within_tolerance():
    """Allow up to 2% regression on a secondary metric per CLAUDE.md §7."""
    champ = _summary({"eval_f1_macro": 0.80, "eval_accuracy": 0.90, "eval_f1_weighted": 0.85})
    chal = _summary(
        {"eval_f1_macro": 0.84, "eval_accuracy": 0.888, "eval_f1_weighted": 0.85}
    )  # acc -1.3%
    d = gate_classifier("seniority", champ, chal)
    assert d.promote is True


def test_returns_compare_objects_for_all_metrics():
    champ = _summary({"eval_f1_macro": 0.80, "eval_accuracy": 0.85, "eval_f1_weighted": 0.82})
    chal = _summary({"eval_f1_macro": 0.83, "eval_accuracy": 0.86, "eval_f1_weighted": 0.83})
    d = gate_classifier("seniority", champ, chal)
    assert d.primary.name == "eval.eval_f1_macro"
    assert {s.name for s in d.secondary} == {"eval.eval_accuracy", "eval.eval_f1_weighted"}
    assert all(s.champion is not None and s.challenger is not None for s in d.secondary)


# ── Lower-is-better (loss-style) ──────────────────────────────────────────


def test_salary_regressor_promotes_on_mae_drop():
    champ = {"test_mae": 30_000, "cv_mae": 31_000, "test_mape": 0.18}
    chal = {"test_mae": 28_500, "cv_mae": 30_000, "test_mape": 0.17}  # -5% MAE
    d = gate_salary_regressor(champ, chal)
    assert d.promote is True


def test_salary_regressor_holds_on_mae_rise():
    champ = {"test_mae": 30_000, "cv_mae": 31_000, "test_mape": 0.18}
    chal = {"test_mae": 30_300, "cv_mae": 31_500, "test_mape": 0.18}  # +1% MAE
    d = gate_salary_regressor(champ, chal)
    assert d.promote is False


def test_salary_regressor_holds_on_secondary_regression():
    champ = {"test_mae": 30_000, "cv_mae": 31_000, "test_mape": 0.18}
    chal = {"test_mae": 28_500, "cv_mae": 33_000, "test_mape": 0.18}  # CV-MAE +6%
    d = gate_salary_regressor(champ, chal)
    assert d.promote is False
    assert "cv_mae" in d.reason


# ── Edge cases ────────────────────────────────────────────────────────────


def test_missing_primary_metric_does_not_promote():
    champ = _summary({"eval_accuracy": 0.85})  # no eval_f1_macro
    chal = _summary({"eval_f1_macro": 0.90, "eval_accuracy": 0.86, "eval_f1_weighted": 0.85})
    d = gate_classifier("seniority", champ, chal)
    assert d.promote is False
    assert "missing" in d.reason


def test_thresholds_are_configurable():
    champ = _summary({"eval_f1_macro": 0.80, "eval_accuracy": 0.85, "eval_f1_weighted": 0.82})
    chal = _summary(
        {"eval_f1_macro": 0.804, "eval_accuracy": 0.85, "eval_f1_weighted": 0.82}
    )  # +0.5%
    d = gate_classifier(
        "seniority",
        champ,
        chal,
        primary_lift_threshold=0.001,  # 0.1% — way below 0.5% lift
    )
    assert d.promote is True


def test_default_thresholds_match_claude_md_spec():
    """CLAUDE.md §7 says '≥1% on primary, no >2% regression on secondary'."""
    assert DEFAULT_PRIMARY_LIFT_THRESHOLD == 0.01
    assert DEFAULT_SECONDARY_REGRESSION_TOLERANCE == 0.02


def test_gate_with_explicit_paths():
    """The lower-level `gate()` function works with arbitrary metric paths."""
    champ = {"foo": {"bar": 0.5}}
    chal = {"foo": {"bar": 0.6}}
    d = gate(
        model="custom",
        champion=champ,
        challenger=chal,
        primary_metric="foo.bar",
        secondary_metrics=[],
    )
    assert d.promote is True
    assert d.primary.delta == pytest.approx(0.1, abs=1e-6)
