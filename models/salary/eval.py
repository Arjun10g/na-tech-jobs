"""Evaluation harness: per-tier headline + stratified metrics + bootstrap CIs.

All predictions arrive on the **log10** scale and are back-transformed to
USD/year before MAE / MAPE are computed (so the units are recruiter-readable).
R² stays on the log scale (per ``LITERATURE_REVIEW.md`` §1.4).

Bootstrap 95% CIs on MAE substitute for the formal power analysis we
deliberately skipped in §15.3 #15.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("models.salary.eval")

DEFAULT_BOOTSTRAP_N: int = 500
DEFAULT_RNG_SEED: int = 42


def _back_transform(y_log: np.ndarray) -> np.ndarray:
    return 10.0 ** np.asarray(y_log)


def mae_usd(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    y_true = _back_transform(y_true_log)
    y_pred = _back_transform(y_pred_log)
    return float(np.mean(np.abs(y_pred - y_true)))


def mape_pct(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    y_true = _back_transform(y_true_log)
    y_pred = _back_transform(y_pred_log)
    return float(np.mean(np.abs(y_pred - y_true) / np.maximum(y_true, 1.0)) * 100)


def r2_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    y_true = np.asarray(y_true_log)
    y_pred = np.asarray(y_pred_log)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def bootstrap_mae_ci(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_RNG_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true_log)
    boot_maes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_maes[i] = mae_usd(y_true_log[idx], y_pred_log[idx])
    return {
        "mae": mae_usd(y_true_log, y_pred_log),
        "mae_ci_low": float(np.quantile(boot_maes, alpha / 2)),
        "mae_ci_high": float(np.quantile(boot_maes, 1 - alpha / 2)),
        "n": int(n),
    }


def stratified_metrics(
    y_true_log: pd.Series,
    y_pred_log: np.ndarray,
    strata: pd.Series,
) -> pd.DataFrame:
    """Per-stratum MAE / MAPE / n with bootstrap MAE CI (no MAPE CI to keep
    output narrow)."""
    df = pd.DataFrame(
        {
            "stratum": strata.values,
            "y_true_log": y_true_log.values,
            "y_pred_log": y_pred_log,
        }
    )
    rows = []
    for stratum, grp in df.groupby("stratum"):
        if len(grp) < 5:
            continue
        ci = bootstrap_mae_ci(grp["y_true_log"].values, grp["y_pred_log"].values)
        rows.append(
            {
                "stratum": stratum,
                "n": len(grp),
                "mae": ci["mae"],
                "mae_ci_low": ci["mae_ci_low"],
                "mae_ci_high": ci["mae_ci_high"],
                "mape_pct": mape_pct(grp["y_true_log"].values, grp["y_pred_log"].values),
                "r2_log": r2_log(grp["y_true_log"].values, grp["y_pred_log"].values),
            }
        )
    if not rows:
        # All strata had fewer than 5 rows — no per-stratum reporting possible.
        return pd.DataFrame(
            columns=["stratum", "n", "mae", "mae_ci_low", "mae_ci_high", "mape_pct", "r2_log"]
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


@dataclass(slots=True)
class TierResult:
    """One tier's evaluation summary."""

    name: str
    mae: float
    mae_ci_low: float
    mae_ci_high: float
    mape_pct: float
    r2_log: float
    n_test: int
    stratified: pd.DataFrame
    extra: dict[str, Any]

    def headline_dict(self) -> dict[str, Any]:
        out = {
            "tier": self.name,
            "mae": round(self.mae, 0),
            "mae_ci_low": round(self.mae_ci_low, 0),
            "mae_ci_high": round(self.mae_ci_high, 0),
            "mape_pct": round(self.mape_pct, 2),
            "r2_log": round(self.r2_log, 4),
            "n_test": self.n_test,
        }
        # Surface CV metrics from extra (when present) into the leaderboard.
        cv_keys = ("cv_mae", "cv_mae_ci_low", "cv_mae_ci_high", "cv_mape_pct", "cv_r2_log")
        for k in cv_keys:
            if k in self.extra:
                v = self.extra[k]
                out[k] = round(v, 4) if k in ("cv_mape_pct", "cv_r2_log") else round(v, 0)
        return out


def evaluate_tier(
    name: str,
    y_true_log: pd.Series,
    y_pred_log: np.ndarray,
    strata: pd.Series,
    extra: dict[str, Any] | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
) -> TierResult:
    overall = bootstrap_mae_ci(y_true_log.values, y_pred_log, n_bootstrap=n_bootstrap)
    return TierResult(
        name=name,
        mae=overall["mae"],
        mae_ci_low=overall["mae_ci_low"],
        mae_ci_high=overall["mae_ci_high"],
        mape_pct=mape_pct(y_true_log.values, y_pred_log),
        r2_log=r2_log(y_true_log.values, y_pred_log),
        n_test=int(len(y_true_log)),
        stratified=stratified_metrics(y_true_log, y_pred_log, strata),
        extra=extra or {},
    )


def leaderboard(results: Iterable[TierResult]) -> pd.DataFrame:
    rows = [r.headline_dict() for r in results]
    return pd.DataFrame(rows).sort_values("mae", ascending=True).reset_index(drop=True)


# ── 5-fold CV-MAE on training data ────────────────────────────────────────


# A ``RefitFn`` takes a training fold + a validation fold (both as their own
# (X, y, strata) triples) and returns predictions on the validation fold's X.
# The encoder is owned by the refit fn so each fold can refit a fresh encoder
# without leaking the validation fold's target into target-encoding fits.
RefitFn = Callable[
    [pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series],
    np.ndarray,
]


def cv_oof_evaluate_tier(
    name: str,
    refit_fn: RefitFn,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strata_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
) -> dict[str, Any]:
    """5-fold OOF evaluation on training data.

    Each fold refits the encoder + model from scratch on the training fold
    and predicts on the held-out validation fold; OOF predictions are
    concatenated and scored with the same MAE / MAPE / R² helpers used for
    test-set evaluation. Stratification is on ``strata_train`` (typically
    ``country/source``) to keep cell representation balanced across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(X_train), np.nan, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, strata_train), start=1):
        X_fold = X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx]
        strata_fold = strata_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        strata_val = strata_train.iloc[val_idx]
        oof[val_idx] = refit_fn(X_fold, y_fold, strata_fold, X_val, strata_val)
        logger.info("CV %s :: fold %d/%d done", name, fold, n_splits)

    if np.isnan(oof).any():
        raise RuntimeError(
            f"OOF predictions for tier {name} contain NaN — refit_fn must "
            "return predictions for every validation row."
        )

    ci = bootstrap_mae_ci(y_train.values, oof, n_bootstrap=n_bootstrap)
    return {
        "cv_mae": ci["mae"],
        "cv_mae_ci_low": ci["mae_ci_low"],
        "cv_mae_ci_high": ci["mae_ci_high"],
        "cv_mape_pct": mape_pct(y_train.values, oof),
        "cv_r2_log": r2_log(y_train.values, oof),
        "cv_n_train": int(len(X_train)),
        "cv_n_splits": n_splits,
    }
