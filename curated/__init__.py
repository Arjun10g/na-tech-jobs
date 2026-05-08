"""Curated layer: deduplicated, history-tracked rolling table over weekly snapshots.

The curated layer is data engineering, not feature engineering — extracted
features from ``ingestion.feature_extraction`` already live on the snapshots.
This module reconciles snapshots over time:
- ``curated/jobs.parquet`` — currently active jobs (in the latest snapshot).
- ``curated/jobs_history.parquet`` — every job ever observed, with
  ``first_seen_at`` and ``last_seen_at`` timestamps.

DuckDB views (``curated/duckdb_views.sql``) sit on top of these for common
analytical queries (active by country, salary-disclosed subset, by role
family, etc.).
"""

from curated.build import build_curated

__all__ = ["build_curated"]
