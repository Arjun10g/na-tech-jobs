"""Tests for ``curated.build``.

Synthesizes 2-3 fake snapshots covering classic dedup scenarios
(continuing job, delisted job, brand-new job) and verifies the active /
history outputs reflect them correctly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from curated.build import build_curated


def _make_row(job_id: str, slug: str = "acme", title: str = "Engineer") -> dict:
    return {
        "id": job_id,
        "company_slug": slug,
        "company_name": slug.capitalize(),
        "title": title,
        "url": f"https://acme.com/jobs/{job_id}",
        "country": "US",
        "salary_disclosed": False,
        "description_md": "Some description.",
        "scraped_at": datetime.now(timezone.utc),
        "source": "greenhouse",
        "raw_payload_hash": "deadbeef",
        "extraction_version": "v1",
    }


def _write_snapshot(tmp: Path, date: str, rows: list[dict]) -> Path:
    snapshot_dir = tmp / date
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], utc=True)
    p = snapshot_dir / "jobs.parquet"
    df.to_parquet(p, index=False)
    return p


def test_single_snapshot(tmp_path: Path):
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    _write_snapshot(snaps, "2026-05-08", [_make_row("a"), _make_row("b")])
    active, history, stats = build_curated(snaps)
    assert len(active) == 2
    assert len(history) == 2
    assert stats["delisted_rows"] == 0
    assert stats["snapshot_count"] == 1
    # first_seen_at == last_seen_at on a single-snapshot world
    for _, row in active.iterrows():
        assert str(row["first_seen_at"])[:10] == "2026-05-08"
        assert str(row["last_seen_at"])[:10] == "2026-05-08"
        assert row["times_seen"] == 1


def test_continuing_job_first_seen_held_constant(tmp_path: Path):
    """A job present in week 1 and week 2 should keep first_seen_at = week 1."""
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    _write_snapshot(snaps, "2026-05-01", [_make_row("a")])
    _write_snapshot(snaps, "2026-05-08", [_make_row("a")])
    active, history, _ = build_curated(snaps)
    assert len(active) == 1
    row = active.iloc[0]
    assert str(row["first_seen_at"])[:10] == "2026-05-01"
    assert str(row["last_seen_at"])[:10] == "2026-05-08"
    assert row["times_seen"] == 2


def test_delisted_job_excluded_from_active_present_in_history(tmp_path: Path):
    """A job in week 1 but not week 2 should be in history but not active."""
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    _write_snapshot(snaps, "2026-05-01", [_make_row("a"), _make_row("b")])
    _write_snapshot(snaps, "2026-05-08", [_make_row("a")])  # b delisted
    active, history, stats = build_curated(snaps)
    active_ids = set(active["id"])
    history_ids = set(history["id"])
    assert active_ids == {"a"}
    assert history_ids == {"a", "b"}
    assert stats["delisted_rows"] == 1
    # The delisted row in history shows last_seen_at = week 1
    delisted_row = history[history["id"] == "b"].iloc[0]
    assert str(delisted_row["last_seen_at"])[:10] == "2026-05-01"


def test_new_job_first_seen_at_latest(tmp_path: Path):
    """A job appearing only in week 2 should have first_seen_at = last_seen_at = week 2."""
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    _write_snapshot(snaps, "2026-05-01", [_make_row("a")])
    _write_snapshot(snaps, "2026-05-08", [_make_row("a"), _make_row("c")])
    active, _, _ = build_curated(snaps)
    new_row = active[active["id"] == "c"].iloc[0]
    assert str(new_row["first_seen_at"])[:10] == "2026-05-08"
    assert str(new_row["last_seen_at"])[:10] == "2026-05-08"


def test_no_snapshots_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_curated(tmp_path / "nonexistent")
