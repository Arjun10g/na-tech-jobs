"""Inference wrapper for the seniority classifier.

Architecture: frozen sentence-transformer embeddings + multinomial logistic
regression (per ``LITERATURE_REVIEW.md`` §17). ``load_from_hub`` pulls the
joblib artifact, re-instantiates the encoder by id, applies the LR head.

Usage::

    from models.seniority.predict import SeniorityClassifier
    clf = SeniorityClassifier.load_from_hub()
    clf.predict(["Senior ML Engineer at Stripe", "Director of Data Science"])
"""

from __future__ import annotations

from models._classifier_inference import LinearProbeClassifier
from models.seniority.train import SPEC


class SeniorityClassifier(LinearProbeClassifier):
    """Thin alias so the public API stays ``SeniorityClassifier.load_from_hub()``."""

    @classmethod
    def load_from_hub(cls, repo_id: str = SPEC.hf_repo_id) -> SeniorityClassifier:
        return super().load_from_hub(repo_id)  # type: ignore[return-value]
