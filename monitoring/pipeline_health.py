"""Pipeline health rollup: per-extractor + per-stage success/failure stats.

Reads JSON snapshots produced by ``ingestion.orchestrator`` (one per
weekly run, written to ``data/snapshots/<date>/ingestion_stats.json``)
and the curated build/enrich stats (``data/curated/build_stats.json``,
``data/curated_enriched/enrich_stats.json``). Surfaces a single-shot
"is the pipeline alive" summary the dashboard tab consumes.

When no ingestion stats exist (fresh checkout, no weekly runs yet), we
fall back to deriving counts directly from the latest snapshot parquet
so the dashboard always has *something* to show.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("monitoring.pipeline_health")


@dataclass
class PipelineHealth:
    """Slim snapshot of last-run state across the whole flywheel."""

    last_ingest_at: str | None = None
    last_ingest_n_jobs: int = 0
    last_ingest_per_extractor: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_curated_build_at: str | None = None
    last_curated_build_n_jobs: int = 0
    last_enrich_at: str | None = None
    last_enrich_n_jobs: int = 0
    last_enrich_coverage: dict[str, int] = field(default_factory=dict)
    snapshots_present: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("malformed JSON :: %s", path)
        return None


def _scan_ingest_stats(snapshots_dir: Path) -> tuple[Path, dict] | tuple[None, None]:
    if not snapshots_dir.exists():
        return None, None
    candidates = sorted(snapshots_dir.glob("*/ingestion_stats.json"))
    if not candidates:
        return None, None
    latest = candidates[-1]
    return latest, _load_json(latest) or {}


def collect_health(
    *,
    snapshots_dir: Path = Path("data/snapshots"),
    curated_path: Path = Path("data/curated/jobs.parquet"),
    curated_stats_path: Path = Path("data/curated/build_stats.json"),
    enrich_stats_path: Path = Path("data/curated_enriched/enrich_stats.json"),
) -> PipelineHealth:
    """Walk the data tree, return a PipelineHealth populated as far as
    available artifacts allow."""
    health = PipelineHealth()
    health.snapshots_present = (
        sorted(p.name for p in snapshots_dir.glob("*") if p.is_dir())
        if snapshots_dir.exists()
        else []
    )

    # Ingestion stats
    ingest_path, ingest = _scan_ingest_stats(snapshots_dir)
    if ingest:
        health.last_ingest_at = ingest.get("finished_at") or ingest.get("started_at")
        health.last_ingest_n_jobs = int(ingest.get("n_jobs", 0))
        # extractors block: {provider: {n_jobs, n_companies, errors[], ...}}
        for k, v in (ingest.get("extractors") or {}).items():
            if isinstance(v, dict):
                health.last_ingest_per_extractor[k] = {
                    "n_jobs": int(v.get("n_jobs", 0)),
                    "n_companies": int(v.get("n_companies", 0)),
                    "n_errors": len(v.get("errors", []) or []),
                }
    elif curated_path.exists():
        # Fallback: derive from the curated parquet so the dashboard
        # isn't empty.
        df = pd.read_parquet(curated_path, columns=["source", "scraped_at"])
        health.last_ingest_n_jobs = len(df)
        if "scraped_at" in df.columns and not df["scraped_at"].empty:
            health.last_ingest_at = str(df["scraped_at"].max())
        if "source" in df.columns:
            for source, group in df.groupby("source"):
                health.last_ingest_per_extractor[str(source)] = {
                    "n_jobs": len(group),
                    "n_companies": 0,
                    "n_errors": 0,
                }
        health.notes.append(
            "no ingestion_stats.json present — counts derived from "
            "curated parquet (per-extractor company counts unavailable)"
        )

    # Curated build stats
    cstats = _load_json(curated_stats_path)
    if cstats:
        health.last_curated_build_at = cstats.get("finished_at") or cstats.get("started_at")
        health.last_curated_build_n_jobs = int(cstats.get("n_jobs", 0))

    # Enrichment stats
    estats = _load_json(enrich_stats_path)
    if estats:
        health.last_enrich_at = estats.get("finished_at") or estats.get("started_at")
        health.last_enrich_n_jobs = int(estats.get("n_rows", 0))
        coverage = estats.get("coverage") or {}
        health.last_enrich_coverage = {k: int(v) for k, v in coverage.items()}

    return health


def to_summary_md(health: PipelineHealth) -> str:
    """Render PipelineHealth as a markdown card for the dashboard tab."""
    lines = ["### Pipeline health"]
    if health.last_ingest_at:
        lines.append(
            f"- **Last ingest**: `{health.last_ingest_at}` — {health.last_ingest_n_jobs:,} jobs"
        )
    else:
        lines.append("- _No ingest stats yet._")

    if health.last_curated_build_at:
        lines.append(
            f"- **Last curated build**: `{health.last_curated_build_at}` — "
            f"{health.last_curated_build_n_jobs:,} active jobs"
        )
    if health.last_enrich_at:
        cov = health.last_enrich_coverage
        cov_str = ", ".join(f"{k}={v:,}" for k, v in cov.items()) if cov else ""
        lines.append(
            f"- **Last enrichment**: `{health.last_enrich_at}` — "
            f"{health.last_enrich_n_jobs:,} rows scored ({cov_str})"
        )
    if health.snapshots_present:
        lines.append(
            f"- **Snapshots on disk**: {len(health.snapshots_present)} "
            f"({health.snapshots_present[0]} → {health.snapshots_present[-1]})"
        )
    if health.last_ingest_per_extractor:
        lines.append("")
        lines.append("**Per-extractor**:")
        lines.append("| Source | Jobs | Companies | Errors |")
        lines.append("|---|---:|---:|---:|")
        for src, stats in sorted(health.last_ingest_per_extractor.items()):
            lines.append(
                f"| `{src}` | {stats.get('n_jobs', 0):,} | "
                f"{stats.get('n_companies', 0)} | {stats.get('n_errors', 0)} |"
            )
    for note in health.notes:
        lines.append(f"\n_Note: {note}_")
    return "\n".join(lines)


def main() -> int:
    """CLI: `uv run python -m monitoring.pipeline_health` prints summary."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--json", action="store_true", help="Output JSON instead of markdown")
    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    health = collect_health()
    if args.json:
        print(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    **health.__dict__,
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(to_summary_md(health))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
