"""Partial tech-stack extractor.

Tier 1 only matches well-known tokens with strict word-boundary matching, so
the false-positive rate stays low. Open-ended skill mining (e.g. "experience
with distributed inference systems") goes to Tier 2 (NuExtract). This module
intentionally stops at the canonical-name list below.
"""

from __future__ import annotations

import re

from ingestion.feature_extraction.confidence import Extraction

# Canonical tech tokens grouped by family. Each value is the canonical name we
# emit; the keys are the patterns we match (case-insensitive, word-bounded).
# Patterns intentionally use either word-bounded multi-char tokens, or
# disambiguating context (e.g. ".js"). Single-letter / 2-letter tokens like
# "R", "Go" need extra context to avoid false positives in prose.
CANONICAL_TECH: dict[str, str] = {
    # Languages — multi-char names are safe with \b; ambiguous shorts need help
    r"\bPython\b": "Python",
    r"\bTypeScript\b": "TypeScript",
    r"\bJavaScript\b": "JavaScript",
    r"\bGolang\b|\b(?:in|with|using)\s+Go\b|\bGo\s+(?:programming|language|developer)\b": "Go",
    r"\bRust\s+(?:programming|language|developer)?\b|\bin\s+Rust\b|\bwith\s+Rust\b": "Rust",
    r"\bC\+\+\b": "C++",
    r"\bC#\b": "C#",
    r"\bScala\b": "Scala",
    r"\bKotlin\b": "Kotlin",
    r"\bSwift\s+(?:programming|developer)?\b|\bin\s+Swift\b": "Swift",
    r"\bRuby\s+on\s+Rails\b|\bRuby\b(?!\s+(?:Tuesday|Murray))": "Ruby",
    r"\bPHP\b": "PHP",
    # Single-letter R: only when context strongly suggests stats.
    r"\b(?:R\s+(?:and\s+Python|programming|language|statistical))\b|\b(?:Python(?:,|\s+and|\s+or)\s+R)\b|\b(?:R/Python)\b": "R",
    r"\bSQL\b": "SQL",
    # Cloud
    r"\bAWS\b": "AWS",
    r"\bAzure\b": "Azure",
    r"\bGCP\b|\bGoogle\s+Cloud(?:\s+Platform)?\b": "GCP",
    # Data / ML frameworks
    r"\bPyTorch\b": "PyTorch",
    r"\bTensorFlow\b": "TensorFlow",
    r"\bscikit-learn\b|\bsklearn\b": "scikit-learn",
    r"\bXGBoost\b|\bLightGBM\b": "XGBoost",
    r"\bHuggingFace\b|\bHugging\s+Face\b": "HuggingFace",
    r"\b(?:Apache\s+)?Spark\b|\bPySpark\b": "Spark",
    r"\bDatabricks\b": "Databricks",
    r"\bSnowflake\b": "Snowflake",
    r"\bdbt\b": "dbt",
    r"\b(?:Apache\s+)?Airflow\b": "Airflow",
    r"\bDagster\b": "Dagster",
    r"\bPrefect\b": "Prefect",
    r"\bKafka\b": "Kafka",
    r"\bRedis\b": "Redis",
    r"\bPostgres(?:QL|ql)?\b": "Postgres",
    r"\bMongoDB\b": "MongoDB",
    r"\bElasticsearch\b|\bElastic\s+Search\b": "Elasticsearch",
    r"\bDuckDB\b": "DuckDB",
    r"\bClickHouse\b": "ClickHouse",
    r"\bBigQuery\b": "BigQuery",
    r"\bRedshift\b": "Redshift",
    r"\bDynamoDB\b": "DynamoDB",
    # ML / AI
    r"\bLLMs?\b|\blarge\s+language\s+models?\b": "LLMs",
    r"\bRAG\b(?:\s+(?:system|pipeline|architecture))?|\bretrieval[- ]augmented\s+generation\b": "RAG",
    r"\bNLP\b|\bnatural\s+language\s+processing\b": "NLP",
    r"\bcomputer\s+vision\b": "Computer Vision",
    r"\breinforcement\s+learning\b": "RL",
    r"\bMLOps\b": "MLOps",
    r"\bdeep\s+learning\b": "Deep Learning",
    # Containers / infra
    r"\bDocker\b": "Docker",
    r"\bKubernetes\b|\bk8s\b": "Kubernetes",
    r"\bTerraform\b": "Terraform",
    r"\bAnsible\b": "Ansible",
    r"\bGitHub\s+Actions\b": "GitHub Actions",
    r"\bGitLab\s+CI\b": "GitLab CI",
    # Web / frontend — ".js" disambiguates names like "react", "next" from prose
    r"\bReact(?:\.js)?(?=\s|,|\.|$)": "React",
    r"\bVue\.js\b": "Vue",
    r"\bSvelte(?:Kit)?\b": "Svelte",
    r"\bNext\.js\b": "Next.js",
    r"\bNode\.js\b|\bNodeJS\b": "Node.js",
    r"\bGraphQL\b": "GraphQL",
    # Notebooks / viz
    r"\bJupyter\b": "Jupyter",
    r"\bpandas\b(?=\s+(?:library|DataFrame|API)|,|\.|;|$)": "pandas",
    r"\bNumPy\b": "NumPy",
    r"\bTableau\b": "Tableau",
    r"\bPower\s*BI\b": "Power BI",
    r"\bLooker\b": "Looker",
    # Experiment / observability
    r"\bMLflow\b": "MLflow",
    r"\bWeights\s+(?:and|&)\s+Biases\b|\bW&B\b|\bWandB\b": "Weights & Biases",
    r"\bDatadog\b": "Datadog",
    r"\bGrafana\b": "Grafana",
    r"\bPrometheus\b": "Prometheus",
}


# Patterns are case-sensitive when the canonical token is uppercase
# (e.g. "AWS"/"GCP"/"SQL"/"NLP"), case-insensitive otherwise.
def _make_pattern(p: str, name: str) -> re.Pattern[str]:
    flags = 0 if name.isupper() or name in {"AWS", "GCP", "GraphQL"} else re.IGNORECASE
    return re.compile(p, flags)


COMPILED_TECH: list[tuple[re.Pattern[str], str]] = [
    (_make_pattern(p, name), name) for p, name in CANONICAL_TECH.items()
]


def run(text: str, title: str = "") -> dict[str, Extraction]:
    if not text and not title:
        return {}
    haystack = (title or "") + "\n" + (text or "")
    found: list[str] = []
    seen_canonical: set[str] = set()
    for pat, canonical in COMPILED_TECH:
        if pat.search(haystack) and canonical not in seen_canonical:
            found.append(canonical)
            seen_canonical.add(canonical)
    if not found:
        return {}
    # Tier 1 confidence reflects token-match precision; Tier 2 (NuExtract) can
    # extend the list with skills phrased as prose ("experience with vector DBs").
    return {
        "tech_stack": Extraction(
            value=found, confidence=0.8, source="regex", rule_id="tech_keywords"
        )
    }
