"""Tests for the skill-extraction taxonomy normalizer.

The NuExtract model itself is not loaded in CI — only the deterministic
``normalize_to_taxonomy`` helper that maps free-form skill strings into
canonical names is exercised here.
"""

from __future__ import annotations

import pytest

from models.skills.predict import SKILL_TAXONOMY, normalize_to_taxonomy


def test_taxonomy_is_non_empty_and_unique():
    assert len(SKILL_TAXONOMY) > 30
    assert len(SKILL_TAXONOMY) == len(set(SKILL_TAXONOMY))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Python", "Python"),
        ("python", "Python"),
        ("PYTHON", "Python"),
        ("py", "Python"),
        ("ts", "TypeScript"),
        ("k8s", "Kubernetes"),
        ("postgresql", "Postgres"),
        ("Google Cloud", "GCP"),
        ("Google Cloud Platform", "GCP"),
        ("retrieval-augmented generation", "RAG"),
        ("large language model", "LLMs"),
        ("ms azure", "Azure"),
        ("torch", "PyTorch"),
    ],
)
def test_normalize_known_aliases(raw, expected):
    out = normalize_to_taxonomy([raw])
    assert out == [expected]


def test_normalize_drops_unknown_tokens():
    out = normalize_to_taxonomy(["Python", "FluentInArgon", "AWS"])
    assert out == ["Python", "AWS"]


def test_normalize_preserves_order_and_dedups():
    out = normalize_to_taxonomy(["Python", "py", "AWS", "Python", "PYTHON"])
    assert out == ["Python", "AWS"]


def test_normalize_handles_non_strings():
    out = normalize_to_taxonomy(["Python", None, 42, "AWS"])  # type: ignore[list-item]
    assert out == ["Python", "AWS"]


def test_normalize_handles_versioned_tokens():
    """Very common in real JDs: 'Python 3.10', 'Python 2'."""
    out = normalize_to_taxonomy(["Python 3", "Python 3.10", "Python 2.7"])
    assert out == ["Python"]


def test_normalize_empty_input():
    assert normalize_to_taxonomy([]) == []
    assert normalize_to_taxonomy([""]) == []
