"""Push a snapshot directory to the HF Dataset repo as a single commit.

Layout pushed to ``arjun10g/na-tech-jobs`` (matches CLAUDE.md §6):

    snapshots/<date>/jobs.parquet
    latest/jobs.parquet
    companies/companies.yaml
    reports/quality/<date>.json
    reports/ingestion/<date>.json
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

logger = logging.getLogger("ingestion.push_to_hub")

DATASET_REPO = os.environ.get("HF_DATASET_REPO", "arjun10g/na-tech-jobs")


def push_snapshot(
    *,
    snapshot_dir: Path,
    quality_path: Path,
    stats_path: Path,
    companies_yaml: Path,
    latest_parquet: Path,
    snapshot_date: str,
    repo_id: str = DATASET_REPO,
    token: str | None = None,
) -> str:
    """Atomic upload of a full snapshot. Returns the commit SHA."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set; cannot push snapshot")

    api = HfApi(token=token)

    operations: list[CommitOperationAdd] = [
        CommitOperationAdd(
            path_in_repo=f"snapshots/{snapshot_date}/jobs.parquet",
            path_or_fileobj=str(snapshot_dir / "jobs.parquet"),
        ),
        CommitOperationAdd(
            path_in_repo="latest/jobs.parquet",
            path_or_fileobj=str(latest_parquet),
        ),
        CommitOperationAdd(
            path_in_repo="companies/companies.yaml",
            path_or_fileobj=str(companies_yaml),
        ),
        CommitOperationAdd(
            path_in_repo=f"reports/quality/{snapshot_date}.json",
            path_or_fileobj=str(quality_path),
        ),
        CommitOperationAdd(
            path_in_repo=f"reports/ingestion/{snapshot_date}.json",
            path_or_fileobj=str(stats_path),
        ),
    ]

    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"ingest: snapshot {snapshot_date}",
    )
    sha = getattr(commit, "oid", None) or getattr(commit, "commit_url", "<unknown>")
    logger.info("pushed snapshot %s to %s :: %s", snapshot_date, repo_id, sha)
    return str(sha)
