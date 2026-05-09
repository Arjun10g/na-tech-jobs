"""Tests for monitoring.pipeline_health — file-system rollup."""

from __future__ import annotations

import json

import pandas as pd

from monitoring.pipeline_health import collect_health, to_summary_md


def _write_parquet(path, rows):
    pd.DataFrame(rows).to_parquet(path)


def test_collect_health_with_full_artifacts(tmp_path):
    snapshots = tmp_path / "snapshots"
    (snapshots / "2026-05-04").mkdir(parents=True)
    (snapshots / "2026-05-08").mkdir(parents=True)
    ingest_stats = {
        "started_at": "2026-05-08T02:00:00Z",
        "finished_at": "2026-05-08T02:14:00Z",
        "n_jobs": 12_500,
        "extractors": {
            "greenhouse": {"n_jobs": 8000, "n_companies": 35, "errors": []},
            "lever": {"n_jobs": 4000, "n_companies": 20, "errors": ["one boom"]},
        },
    }
    (snapshots / "2026-05-08" / "ingestion_stats.json").write_text(json.dumps(ingest_stats))

    curated = tmp_path / "curated.parquet"
    _write_parquet(curated, [{"id": "j1", "source": "greenhouse", "scraped_at": "x"}])
    cstats = tmp_path / "build_stats.json"
    cstats.write_text(
        json.dumps(
            {
                "finished_at": "2026-05-08T03:00:00Z",
                "n_jobs": 12_300,
            }
        )
    )
    estats = tmp_path / "enrich_stats.json"
    estats.write_text(
        json.dumps(
            {
                "finished_at": "2026-05-08T05:00:00Z",
                "n_rows": 12_300,
                "coverage": {"seniority_label": 12_300, "role_family": 12_300},
            }
        )
    )

    health = collect_health(
        snapshots_dir=snapshots,
        curated_path=curated,
        curated_stats_path=cstats,
        enrich_stats_path=estats,
    )
    assert health.last_ingest_n_jobs == 12_500
    assert health.last_ingest_per_extractor["greenhouse"]["n_jobs"] == 8000
    assert health.last_ingest_per_extractor["lever"]["n_errors"] == 1
    assert health.last_curated_build_n_jobs == 12_300
    assert health.last_enrich_n_jobs == 12_300
    assert health.last_enrich_coverage["seniority_label"] == 12_300
    assert health.snapshots_present == ["2026-05-04", "2026-05-08"]
    assert health.notes == []  # No fallback notes when artifacts present.


def test_collect_health_falls_back_to_curated_parquet(tmp_path):
    snapshots = tmp_path / "snapshots"  # Doesn't exist.
    curated = tmp_path / "curated.parquet"
    _write_parquet(
        curated,
        [
            {"id": "j1", "source": "greenhouse", "scraped_at": "2026-05-08T02:00Z"},
            {"id": "j2", "source": "greenhouse", "scraped_at": "2026-05-08T02:01Z"},
            {"id": "j3", "source": "lever", "scraped_at": "2026-05-08T02:00Z"},
        ],
    )

    health = collect_health(
        snapshots_dir=snapshots,
        curated_path=curated,
        curated_stats_path=tmp_path / "_no.json",
        enrich_stats_path=tmp_path / "_no.json",
    )
    assert health.last_ingest_n_jobs == 3
    assert health.last_ingest_per_extractor["greenhouse"]["n_jobs"] == 2
    assert health.last_ingest_per_extractor["lever"]["n_jobs"] == 1
    assert any("derived from" in n for n in health.notes)


def test_collect_health_empty_state(tmp_path):
    health = collect_health(
        snapshots_dir=tmp_path / "no_snapshots",
        curated_path=tmp_path / "no_curated.parquet",
        curated_stats_path=tmp_path / "no_stats.json",
        enrich_stats_path=tmp_path / "no_stats.json",
    )
    assert health.last_ingest_n_jobs == 0
    assert health.last_ingest_per_extractor == {}
    assert health.snapshots_present == []


def test_to_summary_md_renders_without_data():
    from monitoring.pipeline_health import PipelineHealth

    md = to_summary_md(PipelineHealth())
    assert "Pipeline health" in md
    assert "No ingest stats yet" in md


def test_to_summary_md_renders_with_extractors():
    from monitoring.pipeline_health import PipelineHealth

    h = PipelineHealth(
        last_ingest_at="2026-05-08T02:00Z",
        last_ingest_n_jobs=10_000,
        last_ingest_per_extractor={
            "greenhouse": {"n_jobs": 8000, "n_companies": 30, "n_errors": 0},
        },
    )
    md = to_summary_md(h)
    assert "10,000" in md
    assert "greenhouse" in md
    assert "8,000" in md
