"""Lazily fetch the salary regressor + curated parquet from the HF Hub.

Model + data are downloaded on first use and cached under
``~/.cache/huggingface/hub`` (the default for ``huggingface_hub``); on the
HF Spaces runtime that's persistent for the life of the container.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("app.model_loader")

DEFAULT_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "arjun10g/na-tech-jobs-salary-v1")
DEFAULT_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "arjun10g/na-tech-jobs")
PREDICTOR_FILENAME = "salary_predictor.joblib"
CURATED_FILENAME = "curated/jobs.parquet"

_predictor_singleton = None
_curated_path_singleton: Path | None = None


def get_predictor():
    """Return a cached :class:`models.salary.predict.SalaryPredictor`."""
    global _predictor_singleton
    if _predictor_singleton is not None:
        return _predictor_singleton

    from huggingface_hub import hf_hub_download

    from models.salary.predict import SalaryPredictor

    logger.info("downloading salary predictor from %s", DEFAULT_MODEL_REPO)
    path = hf_hub_download(
        repo_id=DEFAULT_MODEL_REPO,
        filename=PREDICTOR_FILENAME,
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),  # private repos / rate-limit relief
    )
    _predictor_singleton = SalaryPredictor.load(Path(path))
    logger.info("salary predictor loaded")
    return _predictor_singleton


def get_curated_path() -> Path:
    """Return the local path of the curated jobs parquet (download once + cache)."""
    global _curated_path_singleton
    if _curated_path_singleton is not None:
        return _curated_path_singleton
    from huggingface_hub import hf_hub_download

    logger.info("downloading curated parquet from %s", DEFAULT_DATASET_REPO)
    path = hf_hub_download(
        repo_id=DEFAULT_DATASET_REPO,
        filename=CURATED_FILENAME,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    _curated_path_singleton = Path(path)
    logger.info("curated parquet at %s", _curated_path_singleton)
    return _curated_path_singleton
