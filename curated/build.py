"""Build the curated layer from weekly snapshots.

DuckDB stacks every ``snapshots/<date>/jobs.parquet`` into a single virtual
table (with each row tagged by its source ``snapshot_date``), then computes
``first_seen_at`` / ``last_seen_at`` per ``id``. The "active" output is the
latest snapshot's rows, with the history columns appended.

Run:
    uv run python -m curated.build --snapshots-dir data/snapshots --output-dir data
    uv run python -m curated.build --push-to-hub
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger("curated.build")


def _stack_snapshots(con: duckdb.DuckDBPyConnection, snapshot_paths: list[Path]) -> str:
    """Register each snapshot as a temp view tagged with its date.

    Returns the SQL fragment that UNIONs them in a single ``stacked`` CTE.
    """
    views: list[str] = []
    for i, path in enumerate(snapshot_paths):
        snapshot_date = path.parent.name  # YYYY-MM-DD
        view_name = f"snap_{i}"
        con.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT *, '{snapshot_date}'::DATE AS snapshot_date "
            f"FROM read_parquet('{path.as_posix()}')"
        )
        views.append(f"SELECT * FROM {view_name}")
    return " UNION ALL BY NAME ".join(views)


def build_curated(snapshots_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read every snapshot under ``snapshots_dir`` and produce ``(active, history, stats)``.

    Each subdirectory of ``snapshots_dir`` named ``YYYY-MM-DD`` is expected to
    contain a ``jobs.parquet``. Subdirectories without that file are skipped
    (e.g. partially-written runs).
    """
    snapshot_paths = sorted(
        p for p in snapshots_dir.glob("*/jobs.parquet") if p.parent.name[:4].isdigit()
    )
    if not snapshot_paths:
        raise FileNotFoundError(f"No snapshots found under {snapshots_dir}")

    snapshot_dates = [p.parent.name for p in snapshot_paths]
    latest_date = snapshot_dates[-1]
    logger.info(
        "stacking %d snapshots (%s … %s)",
        len(snapshot_paths),
        snapshot_dates[0],
        latest_date,
    )

    con = duckdb.connect()
    union_sql = _stack_snapshots(con, snapshot_paths)

    # History: one row per id, with first/last seen across all snapshots
    history_query = f"""
    WITH stacked AS ({union_sql})
    SELECT id,
           MIN(snapshot_date) AS first_seen_at,
           MAX(snapshot_date) AS last_seen_at,
           COUNT(*) AS times_seen
    FROM stacked
    GROUP BY id
    """
    con.execute(f"CREATE OR REPLACE VIEW v_history_keys AS {history_query}")

    # Active: rows in the latest snapshot, joined to first/last seen
    active_df = con.execute(
        f"""
        WITH stacked AS ({union_sql})
        SELECT s.* EXCLUDE (snapshot_date),
               h.first_seen_at,
               h.last_seen_at,
               h.times_seen
        FROM stacked s
        JOIN v_history_keys h USING (id)
        WHERE s.snapshot_date = DATE '{latest_date}'
          AND h.last_seen_at = DATE '{latest_date}'
        """
    ).df()

    # History parquet: every job ever seen (latest observation for each id)
    history_df = con.execute(
        f"""
        WITH stacked AS ({union_sql}),
        ranked AS (
            SELECT s.*,
                   ROW_NUMBER() OVER (PARTITION BY id ORDER BY snapshot_date DESC) AS rn
            FROM stacked s
        )
        SELECT r.* EXCLUDE (rn, snapshot_date),
               h.first_seen_at,
               h.last_seen_at,
               h.times_seen
        FROM ranked r
        JOIN v_history_keys h USING (id)
        WHERE r.rn = 1
        """
    ).df()

    stats = {
        "snapshots": snapshot_dates,
        "snapshot_count": len(snapshot_dates),
        "latest_snapshot": latest_date,
        "active_rows": len(active_df),
        "history_rows": len(history_df),
        "delisted_rows": len(history_df) - len(active_df),
    }
    logger.info(
        "curated build :: active=%d history=%d delisted=%d",
        stats["active_rows"],
        stats["history_rows"],
        stats["delisted_rows"],
    )
    return active_df, history_df, stats


def write_curated(
    active: pd.DataFrame,
    history: pd.DataFrame,
    stats: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    curated_dir = output_dir / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    active_path = curated_dir / "jobs.parquet"
    history_path = curated_dir / "jobs_history.parquet"
    stats_path = curated_dir / "build_stats.json"
    active.to_parquet(active_path, index=False)
    history.to_parquet(history_path, index=False)
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    return {"active": active_path, "history": history_path, "stats": stats_path}


def push_curated_to_hub(paths: dict[str, Path], snapshot_date: str) -> str:
    """Push curated artifacts to the HF Dataset repo."""
    from huggingface_hub import CommitOperationAdd, HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set; cannot push curated layer")
    repo_id = os.environ.get("HF_DATASET_REPO", "arjun10g/na-tech-jobs")
    api = HfApi(token=token)
    operations = [
        CommitOperationAdd(
            path_in_repo="curated/jobs.parquet",
            path_or_fileobj=str(paths["active"]),
        ),
        CommitOperationAdd(
            path_in_repo="curated/jobs_history.parquet",
            path_or_fileobj=str(paths["history"]),
        ),
        CommitOperationAdd(
            path_in_repo=f"curated/build_stats/{snapshot_date}.json",
            path_or_fileobj=str(paths["stats"]),
        ),
    ]
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"curated: rebuild over snapshots up to {snapshot_date}",
    )
    sha = getattr(commit, "oid", None) or "<unknown>"
    logger.info("pushed curated layer to %s :: %s", repo_id, sha)
    return str(sha)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots-dir", default="data/snapshots")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    active, history, stats = build_curated(Path(args.snapshots_dir))
    paths = write_curated(active, history, stats, Path(args.output_dir))
    logger.info(
        "wrote %s, %s, %s",
        paths["active"],
        paths["history"],
        paths["stats"],
    )
    if args.push_to_hub:
        push_curated_to_hub(paths, snapshot_date=stats["latest_snapshot"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
