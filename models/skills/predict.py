"""Skill extraction with NuExtract-tiny + canonical taxonomy normalization.

The taxonomy below is the project's stable skill vocabulary — anything the
model emits gets normalized into one of these canonical names (or dropped).
Add new entries when (a) we see them in real postings repeatedly and
(b) downstream consumers actually use them.

Usage::

    from models.skills.predict import SkillExtractor
    extractor = SkillExtractor()
    skills = extractor.extract(description_md, title="Senior ML Engineer")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("models.skills.predict")

# Canonical taxonomy — same names as in ``ingestion/feature_extraction/regex/tech_stack.py``,
# extended for the skills extractor's broader prompt.
SKILL_TAXONOMY: tuple[str, ...] = (
    # Languages
    "Python",
    "TypeScript",
    "JavaScript",
    "Go",
    "Rust",
    "C++",
    "C#",
    "Scala",
    "Kotlin",
    "Swift",
    "Ruby",
    "PHP",
    "R",
    "SQL",
    "Java",
    # Cloud
    "AWS",
    "Azure",
    "GCP",
    # Data / ML frameworks
    "PyTorch",
    "TensorFlow",
    "scikit-learn",
    "XGBoost",
    "HuggingFace",
    "Spark",
    "Databricks",
    "Snowflake",
    "dbt",
    "Airflow",
    "Dagster",
    "Prefect",
    "Kafka",
    "Redis",
    "Postgres",
    "MongoDB",
    "Elasticsearch",
    "DuckDB",
    "ClickHouse",
    "BigQuery",
    "Redshift",
    "DynamoDB",
    # ML / AI specific
    "LLMs",
    "RAG",
    "NLP",
    "Computer Vision",
    "RL",
    "MLOps",
    "Deep Learning",
    # Containers / infra
    "Docker",
    "Kubernetes",
    "Terraform",
    "Ansible",
    "GitHub Actions",
    "GitLab CI",
    # Web / frontend
    "React",
    "Vue",
    "Svelte",
    "Next.js",
    "Node.js",
    "GraphQL",
    # Notebooks / viz
    "Jupyter",
    "pandas",
    "NumPy",
    "Tableau",
    "Power BI",
    "Looker",
    # Experiment / observability
    "MLflow",
    "Weights & Biases",
    "Datadog",
    "Grafana",
    "Prometheus",
)

# Build a lower-case lookup for fuzzy normalization. Keys = lowercase form,
# values = canonical name.
_TAXONOMY_LOOKUP: dict[str, str] = {s.lower(): s for s in SKILL_TAXONOMY}
_TAXONOMY_LOOKUP.update(
    {
        # Common aliases the model may emit.
        "py": "Python",
        "ts": "TypeScript",
        "js": "JavaScript",
        "golang": "Go",
        "tf": "TensorFlow",
        "torch": "PyTorch",
        "sklearn": "scikit-learn",
        "wandb": "Weights & Biases",
        "w&b": "Weights & Biases",
        "k8s": "Kubernetes",
        "postgresql": "Postgres",
        "mongo": "MongoDB",
        "elastic search": "Elasticsearch",
        "google cloud": "GCP",
        "google cloud platform": "GCP",
        "amazon web services": "AWS",
        "ms azure": "Azure",
        "natural language processing": "NLP",
        "large language model": "LLMs",
        "large language models": "LLMs",
        "retrieval augmented generation": "RAG",
        "retrieval-augmented generation": "RAG",
        "computer-vision": "Computer Vision",
        "ml ops": "MLOps",
        "machine learning operations": "MLOps",
    }
)


def normalize_to_taxonomy(skills: list[str]) -> list[str]:
    """Map free-form skill strings to canonical taxonomy entries; drop misses.

    Order is preserved (so the model's notion of importance survives), and
    each canonical name appears at most once in the output.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in skills:
        if not isinstance(raw, str):
            continue
        key = re.sub(r"\s+", " ", raw.strip().lower()).rstrip(".")
        canonical = _TAXONOMY_LOOKUP.get(key)
        if canonical is None:
            # Try a relaxed startswith match for things like "python 3" → "Python".
            for k, v in _TAXONOMY_LOOKUP.items():
                if key.startswith(k + " ") or key == k:
                    canonical = v
                    break
        if canonical and canonical not in seen:
            out.append(canonical)
            seen.add(canonical)
    return out


@dataclass
class SkillExtractor:
    """Wrapper that runs NuExtract once per description with a fixed schema."""

    _nu: Any = field(default=None, init=False)

    def _ensure_loaded(self) -> bool:
        if self._nu is not None:
            return True
        try:
            from ingestion.feature_extraction.llm.nuextract import NuExtract
        except Exception as exc:  # noqa: BLE001
            logger.warning("NuExtract not importable: %s", exc)
            return False
        nu = NuExtract()
        if not nu._ensure_loaded():
            return False
        self._nu = nu
        return True

    def extract(self, description_md: str, title: str = "") -> list[str]:
        """Return a list of canonical taxonomy skills present in the
        description. Empty list when the model isn't loaded or returns
        nothing."""
        if not self._ensure_loaded() or not description_md:
            return []
        # Reuse the cascade's NuExtract.run() with a tech_stack-only schema.
        result = self._nu.run(
            text=description_md,
            title=title,
            missing_fields=["tech_stack", "industry_experience"],
        )
        skills_extraction = result.get("tech_stack")
        if skills_extraction is None or not skills_extraction.value:
            return []
        return normalize_to_taxonomy(list(skills_extraction.value))

    def extract_batch(
        self,
        items: list[tuple[str, str]],
        *,
        batch_size: int = 8,
    ) -> list[list[str]]:
        """Batched variant — ``items`` are ``(description_md, title)`` tuples."""
        if not self._ensure_loaded() or not items:
            return [[] for _ in items]
        nu_items = [
            (desc or "", title or "", ["tech_stack", "industry_experience"])
            for desc, title in items
        ]
        out: list[list[str]] = []
        for start in range(0, len(nu_items), batch_size):
            chunk = nu_items[start : start + batch_size]
            results = self._nu.run_batch(chunk)
            for r in results:
                ts = r.get("tech_stack")
                if ts is None or not ts.value:
                    out.append([])
                else:
                    out.append(normalize_to_taxonomy(list(ts.value)))
        return out
