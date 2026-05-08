"""Dedup canonical-job lists.

Two layers:

1. **Within-snapshot**: drop duplicate `id`s (a single posting picked up by
   two different ATS handles, or duplicated on the same board).
2. **Across snapshots**: compare `id` set against the prior week's snapshot
   to count new / continuing / delisted postings. The prior snapshot is
   downloaded from the HF Dataset by the orchestrator; this module only
   needs the prior `id` set.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from ingestion.schema import CanonicalJob


def to_dataframe(jobs: Iterable[CanonicalJob]) -> pd.DataFrame:
    """Materialize jobs as a DataFrame with primitive (JSON-compatible) values.

    Using ``mode="json"`` collapses enum members to their string values, which
    is what the Pandera schema expects.
    """
    return pd.DataFrame([j.model_dump(mode="json") for j in jobs])


def dedup_within(jobs: list[CanonicalJob]) -> tuple[list[CanonicalJob], int]:
    """Drop in-snapshot duplicate ids; return (unique_jobs, dropped_count)."""
    seen: set[str] = set()
    out: list[CanonicalJob] = []
    for j in jobs:
        if j.id in seen:
            continue
        seen.add(j.id)
        out.append(j)
    return out, len(jobs) - len(out)


def diff_against_prior(jobs: list[CanonicalJob], prior_ids: set[str]) -> dict[str, Any]:
    """Compute new / continuing / delisted counts vs. the prior snapshot."""
    current_ids = {j.id for j in jobs}
    new_ids = current_ids - prior_ids
    continuing_ids = current_ids & prior_ids
    delisted_ids = prior_ids - current_ids
    return {
        "total": len(current_ids),
        "new": len(new_ids),
        "continuing": len(continuing_ids),
        "delisted": len(delisted_ids),
        "new_ids": new_ids,
        "delisted_ids": delisted_ids,
    }


def load_prior_ids(parquet_path: str | None) -> set[str]:
    """Load `id` column from a prior snapshot parquet. Returns empty set if path is None or missing."""
    if not parquet_path:
        return set()
    try:
        df = pd.read_parquet(parquet_path, columns=["id"])
    except (FileNotFoundError, OSError, ValueError):
        return set()
    return set(df["id"].astype(str))
