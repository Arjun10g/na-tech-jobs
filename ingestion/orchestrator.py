"""End-to-end ingestion driver.

Pipeline:
1. Load `ingestion/companies.yaml`.
2. Spin up an `httpx.AsyncClient` and dispatch each company to the appropriate
   extractor. Per-extractor concurrency is bounded; per-company errors are
   captured.
3. Normalize each extracted job (location, salary, French filter, title-derived
   signals).
4. Drop within-snapshot duplicates, then drop non-NA / French rows.
5. Validate against the canonical Pandera schema; bad rows are dropped and
   counted.
6. Compare to prior snapshot (if available) for new/continuing/delisted counts.
7. Write parquet to `data/snapshots/<date>/jobs.parquet` and a `quality.json`
   report alongside.
8. Optionally push to HF Dataset (controlled by `--push-to-hub`).

Run:
    python -m ingestion.orchestrator --output-dir data
    python -m ingestion.orchestrator --push-to-hub --limit 5  # for a fast smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ingestion.dedup import dedup_within, diff_against_prior, load_prior_ids, to_dataframe
from ingestion.extractors import EXTRACTORS
from ingestion.extractors.base import make_client
from ingestion.normalize import is_likely_french, normalize, utc_now_iso
from ingestion.quality import validate
from ingestion.schema import CanonicalJob, CompanyConfig

logger = logging.getLogger("ingestion.orchestrator")


def load_companies(path: str | Path) -> list[CompanyConfig]:
    raw = yaml.safe_load(Path(path).read_text())
    return [CompanyConfig(**c) for c in raw.get("companies", [])]


async def run_extractors(
    companies: Iterable[CompanyConfig],
    sources: tuple[str, ...] | None = None,
) -> tuple[list[CanonicalJob], list[dict[str, Any]]]:
    """Dispatch the configured companies to the right extractor and gather."""
    by_provider: dict[str, list[CompanyConfig]] = {}
    for c in companies:
        if sources and c.provider not in sources:
            continue
        by_provider.setdefault(c.provider, []).append(c)

    all_jobs: list[CanonicalJob] = []
    all_stats: list[dict[str, Any]] = []
    async with make_client() as client:
        tasks = []
        for provider, batch in by_provider.items():
            extractor_cls = EXTRACTORS.get(provider)
            if extractor_cls is None:
                logger.warning(
                    "no extractor for provider=%s; skipping %d companies", provider, len(batch)
                )
                continue
            extractor = extractor_cls(client)
            tasks.append(extractor.fetch_many(batch))
        for jobs, stats in await asyncio.gather(*tasks):
            all_jobs.extend(jobs)
            all_stats.extend(stats)
    return all_jobs, all_stats


def _filter_na(jobs: list[CanonicalJob]) -> tuple[list[CanonicalJob], int]:
    keep: list[CanonicalJob] = []
    dropped = 0
    for j in jobs:
        if j.country in ("US", "CA"):
            keep.append(j)
        else:
            dropped += 1
    return keep, dropped


def _filter_french(jobs: list[CanonicalJob]) -> tuple[list[CanonicalJob], int]:
    keep: list[CanonicalJob] = []
    dropped = 0
    for j in jobs:
        if is_likely_french(j):
            dropped += 1
            continue
        keep.append(j)
    return keep, dropped


def write_outputs(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    snapshot_date: str,
    quality_report: dict[str, Any],
    pipeline_stats: dict[str, Any],
) -> dict[str, Path]:
    snapshot_dir = output_dir / "snapshots" / snapshot_date
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = snapshot_dir / "jobs.parquet"
    df.to_parquet(parquet_path, index=False)

    quality_dir = output_dir / "reports" / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    quality_path = quality_dir / f"{snapshot_date}.json"
    quality_path.write_text(json.dumps(quality_report, indent=2, default=str))

    stats_dir = output_dir / "reports" / "ingestion"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"{snapshot_date}.json"
    stats_path.write_text(json.dumps(pipeline_stats, indent=2, default=str))

    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(latest_dir / "jobs.parquet", index=False)

    return {
        "parquet": parquet_path,
        "quality": quality_path,
        "stats": stats_path,
        "latest_parquet": latest_dir / "jobs.parquet",
    }


def alert(stats: dict[str, Any], success: bool) -> None:
    """Best-effort Discord notification."""
    try:
        from monitoring import alerts as alerts_mod
    except ImportError:
        return
    fields = {k: v for k, v in stats.items() if not isinstance(v, (list, dict, set))}
    if success:
        alerts_mod.alert_success(f"Ingest snapshot {stats.get('snapshot_date')} ✓", **fields)
    else:
        alerts_mod.alert_failure(f"Ingest snapshot {stats.get('snapshot_date')} ✗", **fields)


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    output_dir = Path(args.output_dir)
    snapshot_date = args.snapshot_date or utc_now_iso()
    started_at = datetime.now(timezone.utc)

    companies = load_companies(args.companies_file)
    if args.limit:
        companies = companies[: args.limit]
    if args.source:
        companies = [c for c in companies if c.provider == args.source]

    logger.info("ingesting %d companies → %s", len(companies), output_dir)

    sources = (args.source,) if args.source else None
    raw_jobs, extractor_stats = await run_extractors(companies, sources=sources)
    logger.info("raw jobs from extractors: %d", len(raw_jobs))

    by_default_country = {c.slug: c.default_country for c in companies}
    normalized = [
        normalize(j, default_country=by_default_country.get(j.company_slug)) for j in raw_jobs
    ]

    deduped, dropped_dup = dedup_within(normalized)
    na_only, dropped_non_na = _filter_na(deduped)
    en_only, dropped_french = _filter_french(na_only)

    df = to_dataframe(en_only)
    valid_df, quality_report = validate(df)

    prior_ids = load_prior_ids(args.prior_snapshot)
    diff = diff_against_prior(en_only, prior_ids)

    finished_at = datetime.now(timezone.utc)
    pipeline_stats = {
        "snapshot_date": snapshot_date,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_sec": (finished_at - started_at).total_seconds(),
        "companies_total": len(companies),
        "companies_succeeded": sum(1 for s in extractor_stats if s.get("error") is None),
        "companies_failed": sum(1 for s in extractor_stats if s.get("error") is not None),
        "raw_jobs": len(raw_jobs),
        "dropped_duplicates": dropped_dup,
        "dropped_non_na": dropped_non_na,
        "dropped_french": dropped_french,
        "rows_after_validation": int(quality_report["valid_rows"]),
        "rows_dropped_by_validation": int(quality_report["dropped_rows"]),
        "new_vs_prior": diff["new"],
        "continuing_vs_prior": diff["continuing"],
        "delisted_vs_prior": diff["delisted"],
        "extractor_stats": extractor_stats,
    }

    paths = write_outputs(
        valid_df,
        output_dir=output_dir,
        snapshot_date=snapshot_date,
        quality_report=quality_report,
        pipeline_stats=pipeline_stats,
    )

    logger.info(
        "snapshot done :: %d rows → %s (took %.1fs)",
        len(valid_df),
        paths["parquet"],
        pipeline_stats["duration_sec"],
    )

    if args.push_to_hub:
        from ingestion.push_to_hub import push_snapshot

        push_snapshot(
            snapshot_dir=output_dir / "snapshots" / snapshot_date,
            quality_path=paths["quality"],
            stats_path=paths["stats"],
            companies_yaml=Path(args.companies_file),
            latest_parquet=paths["latest_parquet"],
            snapshot_date=snapshot_date,
        )

    if args.alert:
        alert(pipeline_stats, success=len(valid_df) > 0)

    return 0 if len(valid_df) > 0 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the weekly ATS ingest pipeline.")
    p.add_argument("--companies-file", default="ingestion/companies.yaml")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--snapshot-date", default=None, help="YYYY-MM-DD (default: today UTC)")
    p.add_argument("--prior-snapshot", default=None, help="Path to a prior parquet for dedup diffs")
    p.add_argument("--source", default=None, help="Filter to one provider (greenhouse|lever|ashby)")
    p.add_argument(
        "--limit", type=int, default=None, help="Truncate companies list (smoke testing)"
    )
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--alert", action="store_true", help="Fire Discord alert on success/failure")
    p.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return p.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
