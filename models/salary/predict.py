"""Inference wrapper for the winning tier — used by curated/enrich.py later
and by anyone calling the HF Model repo via `from_pretrained`.

For Phase 2 v1 the wrapper just re-loads the pickled tier-5 (XGBoost) model
and the fitted full encoder. Phase 4+ will replace this with a sklearn
Pipeline serialized as a single artifact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("models.salary.predict")


@dataclass
class SalaryPredictor:
    encoder: object  # FittedEncoder from models.salary.encode
    model: object  # any tier with a .predict method

    def predict_log_usd_yearly(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = self.encoder.transform(X)
        return self.model.predict(X_enc)

    def predict_usd_yearly(self, X: pd.DataFrame) -> np.ndarray:
        return 10.0 ** self.predict_log_usd_yearly(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"encoder": self.encoder, "model": self.model}, path)
        logger.info("saved SalaryPredictor → %s", path)

    @classmethod
    def load(cls, path: Path) -> SalaryPredictor:
        obj = joblib.load(path)
        return cls(encoder=obj["encoder"], model=obj["model"])
