"""Tier 2 (Mincer OLS) and Tier 3 (Ridge over the full encoder) tiers.

Mincer is fit with statsmodels so we get standard errors and 95% CIs on each
β — the labour-econ-comparable interpretability that motivates the Tier 2
detour.

Ridge is sklearn's; α is selected by 5-fold CV from a log-spaced grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

logger = logging.getLogger("models.salary.linear")


@dataclass
class MincerOLS:
    """Mincer 1974 form: log(salary) ~ yoe + yoe² + edu + country dummies."""

    name: str = "tier2_mincer_ols"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MincerOLS:
        import statsmodels.api as sm  # type: ignore

        X_with_const = sm.add_constant(X.values, has_constant="add")
        self._model = sm.OLS(y.values, X_with_const)
        self.results_ = self._model.fit()
        self.feature_names_ = ["intercept", *X.columns]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import statsmodels.api as sm  # type: ignore

        X_with_const = sm.add_constant(X.values, has_constant="add")
        return self.results_.predict(X_with_const)

    def coefficient_table(self) -> pd.DataFrame:
        """β estimates with 95% CIs and p-values."""
        params = self.results_.params
        ci = self.results_.conf_int()
        pvalues = self.results_.pvalues
        df = pd.DataFrame(
            {
                "coef": params,
                "ci_low": ci[:, 0],
                "ci_high": ci[:, 1],
                "p_value": pvalues,
            },
            index=self.feature_names_[: len(params)],
        )
        return df.round(5)


@dataclass
class RidgeOverFull:
    """Tier 3: RidgeCV over the full ColumnTransformer encoder."""

    name: str = "tier3_ridge_full"
    alphas: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeOverFull:
        self.model_ = RidgeCV(alphas=list(self.alphas), cv=5)
        self.model_.fit(X.values, y.values)
        self.alpha_ = float(self.model_.alpha_)
        logger.info("RidgeCV chose α=%s", self.alpha_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.values)

    def coefficient_summary(self, X_columns: list[str], top_k: int = 15) -> pd.DataFrame:
        coefs = pd.Series(self.model_.coef_, index=X_columns)
        return (
            coefs.abs()
            .sort_values(ascending=False)
            .head(top_k)
            .to_frame("abs_coef")
            .join(coefs.rename("coef"))
        )
