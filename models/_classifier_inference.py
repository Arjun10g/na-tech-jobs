"""Shared inference wrapper for the v1 classifiers.

Loads the joblib artifact written by ``models._classifier_base.train_classifier``
(``{encoder_id, classifier, label2id, id2label, ...}``), re-instantiates the
sentence-transformer encoder by id, and exposes ``predict`` / ``predict_proba``.

Used by both ``models.seniority.predict`` and ``models.role_family.predict``
to avoid duplicating the loading / encoding logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from models._classifier_base import ARTIFACT_FILENAME, encode_texts

logger = logging.getLogger("models.classifier_inference")


@dataclass
class LinearProbeClassifier:
    encoder_id: str
    classifier: Any  # sklearn LogisticRegression
    id2label: dict[int, str]
    label2id: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, model_dir: Path) -> LinearProbeClassifier:
        import joblib

        artifact_path = model_dir / ARTIFACT_FILENAME
        if not artifact_path.exists():
            raise FileNotFoundError(f"missing classifier artifact at {artifact_path}")
        artifact = joblib.load(artifact_path)
        return cls(
            encoder_id=artifact["encoder_id"],
            classifier=artifact["classifier"],
            id2label={int(k): v for k, v in artifact["id2label"].items()},
            label2id=artifact["label2id"],
            metadata={
                k: v
                for k, v in artifact.items()
                if k not in {"encoder_id", "classifier", "id2label", "label2id"}
            },
        )

    @classmethod
    def load_from_hub(cls, repo_id: str) -> LinearProbeClassifier:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_id=repo_id, repo_type="model")
        return cls.load(Path(local_dir))

    def _embed(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return encode_texts(texts, self.encoder_id, batch_size=batch_size)

    def predict(self, texts: list[str], *, batch_size: int = 64) -> list[str]:
        if not texts:
            return []
        X = self._embed(texts, batch_size=batch_size)
        ids = self.classifier.predict(X)
        return [self.id2label[int(i)] for i in ids]

    def predict_proba(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.id2label)), dtype=np.float32)
        X = self._embed(texts, batch_size=batch_size)
        return np.asarray(self.classifier.predict_proba(X), dtype=np.float32)
