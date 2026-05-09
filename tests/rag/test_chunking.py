"""Tests for rag.chunking — parent-child recursive split."""

from __future__ import annotations

import pytest

from rag.chunking import (
    PAYLOAD_FIELDS,
    ChildChunk,
    ParentChunk,
    chunk_job,
    chunk_jobs,
)


def _make_row(**overrides):
    base = {
        "id": "abc123",
        "title": "Senior ML Engineer",
        "company_name": "Acme",
        "company_slug": "acme",
        "country": "US",
        "url": "https://example.com/jobs/1",
        "remote_policy": "hybrid",
        "source": "greenhouse",
        "description_md": "## About\n\nWe're hiring.\n\n## Responsibilities\n\nBuild ML.",
        "seniority_extracted": "senior",
        "role_family_extracted": "MLE",
        "salary_min_usd_yearly": 180_000.0,
        "salary_max_usd_yearly": 240_000.0,
        "salary_disclosed": True,
    }
    base.update(overrides)
    return base


def test_chunk_job_returns_at_least_one_parent_and_child():
    parents, children = chunk_job(_make_row())
    assert len(parents) >= 1
    assert len(children) >= 1
    assert all(isinstance(p, ParentChunk) for p in parents)
    assert all(isinstance(c, ChildChunk) for c in children)


def test_chunk_job_ids_are_stable_and_well_formed():
    parents, children = chunk_job(_make_row())
    assert all(p.parent_chunk_id.startswith("abc123::p") for p in parents)
    assert all(c.child_chunk_id.startswith("abc123::c") for c in children)
    assert all(c.parent_chunk_id in {p.parent_chunk_id for p in parents} for c in children)


def test_chunk_job_prepends_title_to_text():
    parents, _ = chunk_job(_make_row(title="Plain Title"))
    assert parents[0].text.startswith("# Plain Title")


def test_chunk_job_handles_empty_description():
    parents, children = chunk_job(_make_row(description_md=""))
    # Title alone still produces one chunk.
    assert len(parents) == 1
    assert len(children) == 1
    assert "Senior ML Engineer" in parents[0].text


def test_chunk_job_returns_empty_when_title_and_desc_both_missing():
    parents, children = chunk_job(_make_row(title="", description_md=""))
    assert parents == []
    assert children == []


def test_chunk_job_payload_only_contains_known_fields():
    row = _make_row(rogue_field="should-be-dropped", _internal=42)
    parents, _ = chunk_job(row)
    payload = parents[0].payload
    assert "rogue_field" not in payload
    assert "_internal" not in payload
    assert payload["id"] == "abc123"
    assert payload["title"] == "Senior ML Engineer"
    assert payload["country"] == "US"


def test_chunk_job_payload_carries_phase4_predictions_when_present():
    row = _make_row(
        seniority_label_v1="senior",
        seniority_confidence_v1=0.91,
        role_family_v1="MLE",
        predicted_salary_usd_v1=215_000.0,
        extracted_skills_v1=["Python", "PyTorch"],
        prediction_model_version="v1",
    )
    parents, _ = chunk_job(row)
    p = parents[0].payload
    assert p["seniority_label_v1"] == "senior"
    assert p["role_family_v1"] == "MLE"
    assert p["predicted_salary_usd_v1"] == 215_000.0
    assert p["extracted_skills_v1"] == ["Python", "PyTorch"]


def test_chunk_jobs_aggregates_across_rows():
    rows = [_make_row(id="a"), _make_row(id="b")]
    parents, children = chunk_jobs(rows)
    job_ids_p = {p.job_id for p in parents}
    job_ids_c = {c.job_id for c in children}
    assert job_ids_p == {"a", "b"}
    assert job_ids_c == {"a", "b"}


def test_payload_fields_includes_filtering_columns():
    """Fields CLAUDE.md §8 wants for Qdrant filters must be in the payload."""
    required = {
        "country",
        "seniority_label_v1",
        "predicted_salary_usd_v1",
        "posted_at",
        "remote_policy",
        "salary_max_usd_yearly",
    }
    assert required.issubset(set(PAYLOAD_FIELDS))


def test_chunk_job_long_description_yields_multiple_parents():
    long_body = "This is a paragraph. " * 200  # ~4k chars > 1024 tokens
    parents, _ = chunk_job(_make_row(description_md=long_body))
    assert len(parents) >= 2


def test_chunk_job_children_are_smaller_than_parents():
    long_body = "This is a paragraph. " * 200
    parents, children = chunk_job(_make_row(description_md=long_body))
    # Average child size should be less than average parent size.
    avg_parent = sum(len(p.text) for p in parents) / len(parents)
    avg_child = sum(len(c.text) for c in children) / len(children)
    assert avg_child < avg_parent


@pytest.mark.parametrize("field_name", ["company_slug", "title", "url", "country", "id"])
def test_payload_keeps_basic_metadata(field_name):
    parents, _ = chunk_job(_make_row())
    assert field_name in parents[0].payload
