"""Tier 0 (constant) and Tier 1 (stratified mean) baselines.

These are not real ML; their job is to set the floor against which Tier 2-5
must demonstrably improve. See ``LITERATURE_REVIEW.md`` §16.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConstantBaseline:
    """Tier 0: predict the mean of the training log-target for every row."""

    name: str = "tier0_constant"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ConstantBaseline:
        self.mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


@dataclass
class StratifiedMeanBaseline:
    """Tier 1: per-stratum mean with empirical-Bayes shrinkage to the global mean.

    Shrinkage formula (cf. ``LITERATURE_REVIEW.md`` §5.2):
        encoded(stratum) = (n*mean(stratum) + m*global_mean) / (n + m)

    The smoothing constant ``m`` defaults to 30 (a sensible NA-tech-corpus
    prior that pulls strata with <30 rows toward the global mean).
    """

    smoothing: float = 30.0
    name: str = "tier1_stratified_mean"

    def fit(self, X: pd.DataFrame, y: pd.Series, *, strata: pd.Series) -> StratifiedMeanBaseline:
        df = pd.DataFrame({"stratum": strata.values, "y": y.values})
        global_mean = float(df["y"].mean())
        agg = df.groupby("stratum")["y"].agg(["mean", "count"])
        smoothed = (agg["count"] * agg["mean"] + self.smoothing * global_mean) / (
            agg["count"] + self.smoothing
        )
        self.global_mean_ = global_mean
        self.encoding_ = smoothed.to_dict()
        return self

    def predict(self, X: pd.DataFrame, *, strata: pd.Series) -> np.ndarray:
        return np.array([self.encoding_.get(s, self.global_mean_) for s in strata])
