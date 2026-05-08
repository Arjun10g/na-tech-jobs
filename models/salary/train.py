"""Train all six tiers, log to MLflow, write a leaderboard.

Run::

    uv run python -m models.salary.train
    uv run python -m models.salary.train --skip-xgboost   # for fast iteration
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import pandas as pd

from models.salary.baselines import ConstantBaseline, StratifiedMeanBaseline
from models.salary.dataset import build_dataset
from models.salary.encode import fit_full_encoder, fit_mincer_encoder
from models.salary.eval import (
    TierResult,
    cv_oof_evaluate_tier,
    evaluate_tier,
    leaderboard,
)
from models.salary.forest import RandomForest
from models.salary.linear import MincerOLS, RidgeOverFull
from models.salary.model_card import write_model_card
from models.salary.predict import SalaryPredictor
from models.salary.xgb import XGBoostOptuna

logger = logging.getLogger("models.salary.train")

DEFAULT_OUTPUT_DIR = Path("data/models/salary")


# ── Per-tier refit functions for CV evaluation ────────────────────────────
# Each takes ``(X_fold, y_fold, strata_fold, X_val, strata_val)`` and returns
# predictions for ``X_val``. Encoders are refit per-fold to avoid target
# encoding leaking the validation fold's targets. For Tier 5 the closure
# captures Optuna's already-tuned best params so CV doesn't re-search.


def _refit_tier0(X_fold, y_fold, strata_fold, X_val, strata_val):
    return ConstantBaseline().fit(X_fold, y_fold).predict(X_val)


def _refit_tier1(X_fold, y_fold, strata_fold, X_val, strata_val):
    return (
        StratifiedMeanBaseline()
        .fit(X_fold, y_fold, strata=strata_fold)
        .predict(X_val, strata=strata_val)
    )


def _refit_tier2(X_fold, y_fold, strata_fold, X_val, strata_val):
    enc = fit_mincer_encoder(X_fold, y_fold)
    Xtr = enc.transform(X_fold)
    Xv = enc.transform(X_val)
    return MincerOLS().fit(Xtr, y_fold).predict(Xv)


def _refit_tier3(X_fold, y_fold, strata_fold, X_val, strata_val):
    enc = fit_full_encoder(X_fold, y_fold)
    Xtr = enc.transform(X_fold)
    Xv = enc.transform(X_val)
    return RidgeOverFull().fit(Xtr, y_fold).predict(Xv)


def _refit_tier4(X_fold, y_fold, strata_fold, X_val, strata_val):
    enc = fit_full_encoder(X_fold, y_fold)
    Xtr = enc.transform(X_fold)
    Xv = enc.transform(X_val)
    return RandomForest().fit(Xtr, y_fold).predict(Xv)


def _make_refit_tier5(best_params: dict):
    """Returns a refit fn that uses already-tuned XGB hyperparams; no new search."""
    import xgboost as xgb

    def _fn(X_fold, y_fold, strata_fold, X_val, strata_val):
        enc = fit_full_encoder(X_fold, y_fold)
        Xtr = enc.transform(X_fold).values
        Xv = enc.transform(X_val).values
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
            **best_params,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(Xtr, y_fold.values)
        return model.predict(Xv)

    return _fn


def _set_mlflow(experiment: str = "salary_regressor") -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def _log_tier(result: TierResult) -> None:
    """Log a finished tier as a child run inside the parent ladder run."""
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.log_params({"tier": result.name})
        mlflow.log_params(result.extra.get("hyperparams", {}))
        mlflow.log_metric("mae_usd", result.mae)
        mlflow.log_metric("mae_ci_low", result.mae_ci_low)
        mlflow.log_metric("mae_ci_high", result.mae_ci_high)
        mlflow.log_metric("mape_pct", result.mape_pct)
        mlflow.log_metric("r2_log", result.r2_log)
        mlflow.log_metric("n_test", result.n_test)
        for cv_key in ("cv_mae", "cv_mae_ci_low", "cv_mae_ci_high", "cv_mape_pct", "cv_r2_log"):
            if cv_key in result.extra:
                mlflow.log_metric(cv_key, float(result.extra[cv_key]))


def _attach_cv(result: TierResult, refit_fn, ds, *, n_splits: int = 5) -> TierResult:
    """Run 5-fold CV-OOF evaluation on the training set and merge the metrics
    into ``result.extra`` so they roundtrip through ``headline_dict``."""
    cv = cv_oof_evaluate_tier(
        result.name,
        refit_fn,
        X_train=ds.X_train,
        y_train=ds.y_train,
        strata_train=ds.strata_train,
        n_splits=n_splits,
    )
    result.extra.update(cv)
    logger.info(
        "%s :: test_mae=%.0f cv_mae=%.0f",
        result.name,
        result.mae,
        cv["cv_mae"],
    )
    return result


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    skip_xgboost: bool = False,
    xgboost_trials: int = 50,
) -> dict:
    _set_mlflow()
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = build_dataset()
    results: list[TierResult] = []

    with mlflow.start_run(run_name="ladder_v1") as parent:
        parent_run_id = parent.info.run_id
        mlflow.log_params(
            {
                "n_train": ds.info["n_train"],
                "n_test": ds.info["n_test"],
                "winsorize_quantile": ds.info["winsorize_quantile"],
                "min_plausible_usd": ds.info["min_plausible_usd"],
            }
        )

        # ── Tier 0 ─────────────────────────────────────────────────────────
        logger.info("Tier 0 — constant baseline")
        m0 = ConstantBaseline().fit(ds.X_train, ds.y_train)
        preds = m0.predict(ds.X_test)
        r0 = evaluate_tier("tier0_constant", ds.y_test, preds, ds.strata_test)
        _attach_cv(r0, _refit_tier0, ds)
        results.append(r0)
        _log_tier(r0)

        # ── Tier 1 ─────────────────────────────────────────────────────────
        logger.info("Tier 1 — stratified-mean baseline")
        m1 = StratifiedMeanBaseline().fit(ds.X_train, ds.y_train, strata=ds.strata_train)
        preds = m1.predict(ds.X_test, strata=ds.strata_test)
        r1 = evaluate_tier("tier1_stratified_mean", ds.y_test, preds, ds.strata_test)
        _attach_cv(r1, _refit_tier1, ds)
        results.append(r1)
        _log_tier(r1)

        # ── Tier 2: Mincer OLS ─────────────────────────────────────────────
        logger.info("Tier 2 — Mincer OLS")
        mincer_enc = fit_mincer_encoder(ds.X_train, ds.y_train)
        Xtr_m, Xte_m = mincer_enc.transform(ds.X_train), mincer_enc.transform(ds.X_test)
        m2 = MincerOLS().fit(Xtr_m, ds.y_train)
        preds = m2.predict(Xte_m)
        coef_table = m2.coefficient_table()
        coef_path = output_dir / "tier2_mincer_coefficients.csv"
        coef_table.to_csv(coef_path)
        r2 = evaluate_tier(
            "tier2_mincer_ols",
            ds.y_test,
            preds,
            ds.strata_test,
            extra={
                "n_features": len(mincer_enc.feature_names),
                "coefficients_path": str(coef_path),
            },
        )
        _attach_cv(r2, _refit_tier2, ds)
        results.append(r2)
        _log_tier(r2)
        mlflow.log_artifact(str(coef_path))

        # ── Tier 3: Ridge ──────────────────────────────────────────────────
        logger.info("Tier 3 — Ridge over full encoder")
        full_enc = fit_full_encoder(ds.X_train, ds.y_train)
        Xtr_full, Xte_full = full_enc.transform(ds.X_train), full_enc.transform(ds.X_test)
        m3 = RidgeOverFull().fit(Xtr_full, ds.y_train)
        preds = m3.predict(Xte_full)
        coef_summary = m3.coefficient_summary(full_enc.feature_names, top_k=20)
        coef_path_3 = output_dir / "tier3_ridge_top_coefs.csv"
        coef_summary.to_csv(coef_path_3)
        r3 = evaluate_tier(
            "tier3_ridge_full",
            ds.y_test,
            preds,
            ds.strata_test,
            extra={
                "n_features": len(full_enc.feature_names),
                "alpha": m3.alpha_,
                "coefficients_path": str(coef_path_3),
            },
        )
        _attach_cv(r3, _refit_tier3, ds)
        results.append(r3)
        _log_tier(r3)
        mlflow.log_artifact(str(coef_path_3))

        # ── Tier 4: Random Forest ──────────────────────────────────────────
        logger.info("Tier 4 — Random Forest")
        m4 = RandomForest().fit(Xtr_full, ds.y_train)
        preds = m4.predict(Xte_full)
        importance_4 = m4.importance(full_enc.feature_names, top_k=20)
        imp_path_4 = output_dir / "tier4_rf_importance.csv"
        importance_4.to_csv(imp_path_4)
        r4 = evaluate_tier(
            "tier4_random_forest",
            ds.y_test,
            preds,
            ds.strata_test,
            extra={
                "hyperparams": {
                    "n_estimators": m4.n_estimators,
                    "min_samples_leaf": m4.min_samples_leaf,
                },
                "importance_path": str(imp_path_4),
            },
        )
        _attach_cv(r4, _refit_tier4, ds)
        results.append(r4)
        _log_tier(r4)
        mlflow.log_artifact(str(imp_path_4))

        # ── Tier 5: XGBoost + Optuna ───────────────────────────────────────
        if not skip_xgboost:
            logger.info("Tier 5 — XGBoost + Optuna (%d trials)", xgboost_trials)
            m5 = XGBoostOptuna(n_trials=xgboost_trials).fit(Xtr_full, ds.y_train)
            preds = m5.predict(Xte_full)
            importance_5 = m5.importance(full_enc.feature_names, top_k=20)
            imp_path_5 = output_dir / "tier5_xgb_importance.csv"
            importance_5.to_csv(imp_path_5)
            best_path = output_dir / "tier5_xgb_best_params.json"
            best_path.write_text(json.dumps(m5.best_params_, indent=2))
            r5 = evaluate_tier(
                "tier5_xgboost_optuna",
                ds.y_test,
                preds,
                ds.strata_test,
                extra={
                    "hyperparams": m5.best_params_,
                    "cv_score": m5.cv_score_,
                    "n_trials": xgboost_trials,
                    "importance_path": str(imp_path_5),
                    "best_params_path": str(best_path),
                },
            )
            _attach_cv(r5, _make_refit_tier5(m5.best_params_), ds)
            results.append(r5)
            _log_tier(r5)
            mlflow.log_artifact(str(imp_path_5))
            mlflow.log_artifact(str(best_path))

        # ── Leaderboard ────────────────────────────────────────────────────
        lb = leaderboard(results)
        lb_path = output_dir / "leaderboard.csv"
        lb.to_csv(lb_path, index=False)
        mlflow.log_artifact(str(lb_path))

        # Stratified breakdowns
        for r in results:
            sp = output_dir / f"{r.name}_stratified.csv"
            r.stratified.to_csv(sp, index=False)
            mlflow.log_artifact(str(sp))

        report = {
            "parent_run_id": parent_run_id,
            "n_train": ds.info["n_train"],
            "n_test": ds.info["n_test"],
            "tiers": [r.headline_dict() for r in results],
            "winning_tier": min(results, key=lambda r: r.mae).name,
        }
        report_path = output_dir / "ladder_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))

        # Persist the winning predictor + auto-generated model card.
        winning_result = next(r for r in results if r.name == report["winning_tier"])
        winning_model = {
            "tier0_constant": m0,
            "tier1_stratified_mean": m1,
            "tier2_mincer_ols": m2,
            "tier3_ridge_full": m3,
            "tier4_random_forest": m4,
            "tier5_xgboost_optuna": m5 if not skip_xgboost else None,
        }[report["winning_tier"]]
        winning_encoder = mincer_enc if report["winning_tier"] == "tier2_mincer_ols" else full_enc
        if report["winning_tier"] not in {"tier0_constant", "tier1_stratified_mean"}:
            predictor_path = output_dir / "salary_predictor.joblib"
            SalaryPredictor(encoder=winning_encoder, model=winning_model).save(predictor_path)
            mlflow.log_artifact(str(predictor_path))

        card_path = output_dir / "MODEL_CARD.md"
        write_model_card(
            report=report,
            stratified=winning_result.stratified,
            out_path=card_path,
        )
        mlflow.log_artifact(str(card_path))

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--skip-xgboost", action="store_true")
    p.add_argument("--xgboost-trials", type=int, default=50)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    report = run(
        output_dir=Path(args.output_dir),
        skip_xgboost=args.skip_xgboost,
        xgboost_trials=args.xgboost_trials,
    )

    print("\n=== Leaderboard ===")
    lb = pd.DataFrame(report["tiers"])
    print(lb.to_string(index=False))
    print(f"\nWinning tier: {report['winning_tier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
