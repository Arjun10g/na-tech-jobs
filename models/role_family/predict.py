"""Inference wrapper for the role-family classifier.

Architecture: frozen sentence-transformer embeddings + multinomial logistic
regression (per ``LITERATURE_REVIEW.md`` §17).
"""

from __future__ import annotations

from models._classifier_inference import LinearProbeClassifier
from models.role_family.train import SPEC


class RoleFamilyClassifier(LinearProbeClassifier):
    @classmethod
    def load_from_hub(cls, repo_id: str = SPEC.hf_repo_id) -> RoleFamilyClassifier:
        return super().load_from_hub(repo_id)  # type: ignore[return-value]
