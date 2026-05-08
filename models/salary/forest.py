"""Tier 4: Random Forest regressor over the full encoder.

Hyperparameters chosen to be close to the off-the-shelf default in
``sklearn`` documentation — Tier 5 (XGBoost + Optuna) is the place where
we actually search. Random Forest is here as a robustness check: tree-based
without boosting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class RandomForest:
    name: str = "tier4_random_forest"
    n_estimators: int = 500
    max_features: str = "sqrt"
    max_depth: int | None = None
    min_samples_leaf: int = 5
    n_jobs: int = -1
    random_state: int = 42

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomForest:
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.model_.fit(X.values, y.values)
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
