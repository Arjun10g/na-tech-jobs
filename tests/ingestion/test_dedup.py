"""Tests for ingestion.dedup."""

from __future__ import annotations

from datetime import datetime, timezone

from ingestion.dedup import dedup_within, diff_against_prior
from ingestion.schema import CanonicalJob


def make_job(idx: int) -> CanonicalJob:
    return CanonicalJob(
        id=f"{idx:016x}",
        company_slug="acme",
        company_name="Acme",
        title=f"Job {idx}",
        url=f"https://acme.com/jobs/{idx}",
        scraped_at=datetime.now(timezone.utc),
        source="greenhouse",
        raw_payload_hash="deadbeef",
    )


def test_dedup_within_drops_repeat_ids():
    a, b, c = make_job(1), make_job(2), make_job(1)
    out, dropped = dedup_within([a, b, c])
    assert len(out) == 2
    assert dropped == 1
    assert {j.id for j in out} == {a.id, b.id}


def test_dedup_within_preserves_order():
    jobs = [make_job(i) for i in range(5)]
    out, _ = dedup_within(jobs)
    assert [j.id for j in out] == [j.id for j in jobs]


def test_diff_against_prior_counts():
    current = [make_job(1), make_job(2), make_job(3)]
    prior_ids = {make_job(2).id, make_job(3).id, make_job(4).id}
    diff = diff_against_prior(current, prior_ids)
    assert diff["total"] == 3
    assert diff["new"] == 1
    assert diff["continuing"] == 2
    assert diff["delisted"] == 1
    assert diff["new_ids"] == {make_job(1).id}
    assert diff["delisted_ids"] == {make_job(4).id}


def test_diff_against_empty_prior():
    current = [make_job(1), make_job(2)]
    diff = diff_against_prior(current, set())
    assert diff["new"] == 2
    assert diff["continuing"] == 0
    assert diff["delisted"] == 0
