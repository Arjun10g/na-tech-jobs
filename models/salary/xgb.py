"""Tier 5: XGBoost regressor with Optuna hyperparameter search.

50-trial search per CLAUDE.md §7; MSE loss on the log-target. The search
space follows the recommendations in [Chen & Guestrin 2016] and the
sensible defaults in ``Hands-On ML 3e`` ch. 7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

logger = logging.getLogger("models.salary.xgb")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class XGBoostOptuna:
    name: str = "tier5_xgboost_optuna"
    n_trials: int = 50
    n_splits: int = 5
    random_state: int = 42

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        params: dict[str, Any] = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": self.random_state,
            "verbosity": 0,
        }
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        cv_mse = []
        for train_idx, val_idx in kf.split(X):
            model = xgb.XGBRegressor(**params)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            cv_mse.append(float(np.mean((preds - y[val_idx]) ** 2)))
        return float(np.mean(cv_mse))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostOptuna:
        X_arr = X.values
        y_arr = y.values
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study_ = optuna.create_study(direction="minimize", sampler=sampler)

        def _obj(trial):
            return self._objective(trial, X_arr, y_arr)

        self.study_.optimize(_obj, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params_ = dict(self.study_.best_params)
        logger.info("Optuna best CV-MSE=%.4f at %s", self.study_.best_value, self.best_params_)

        # Refit on full training data with the best params.
        full_params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": self.random_state,
            "verbosity": 0,
            **self.best_params_,
        }
        self.model_ = xgb.XGBRegressor(**full_params)
        self.model_.fit(X_arr, y_arr)
        self.cv_score_ = float(self.study_.best_value)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.values)

    def importance(self, X_columns: list[str], top_k: int = 15) -> pd.DataFrame:
        return (
            pd.Series(self.model_.feature_importances_, index=X_columns)
            .sort_values(ascending=False)
            .head(top_k)
            .to_frame("importance")
        )
