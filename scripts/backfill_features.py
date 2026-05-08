"""One-shot script: re-run feature extraction over an existing snapshot parquet.

Use after the cascade gains a new extractor or a regex bug fix lands —
re-fetching from the ATS isn't necessary because we preserved
``description_md``. The script:

1. Loads ``data/latest/jobs.parquet`` (or any path passed via ``--input``).
2. Runs HTML→MD on every description (idempotent — markdownify on plain
   markdown is a no-op apart from minor whitespace changes).
3. Calls the feature cascade.
4. Updates ALL extracted-feature columns + ``salary_*``,
   ``remote_policy``, ``extraction_meta``, ``extraction_version`` in place.
5. Re-validates with Pandera; reports row counts + per-feature hit rates.
6. Writes back to a new snapshot directory if ``--snapshot-date`` differs;
   otherwise overwrites in place.
7. Optional ``--push-to-hub`` mirrors the orchestrator behaviour.

    uv run python -m scripts.backfill_features
    uv run python -m scripts.backfill_features --push-to-hub
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from markdownify import markdownify

from ingestion.feature_extraction import extract_features
from ingestion.normalize import (
    extract_role_family,
    extract_seniority,
    normalize_salary,
)

logger = logging.getLogger("backfill")

# Columns the cascade may overwrite. Anything outside this set is preserved.
EXTRACTED_FEATURE_COLS = (
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
    "salary_disclosed",
    "salary_min_usd_yearly",
    "salary_max_usd_yearly",
    "remote_policy",
    "min_years_experience",
    "max_years_experience",
    "min_education",
    "requires_security_clearance",
    "clearance_level",
    "requires_citizenship",
    "offers_visa_sponsorship",
    "offers_relocation",
    "offers_equity",
    "equity_form",
    "bonus_mentioned",
    "bonus_type",
    "max_travel_percent",
    "contract_type",
    "on_call_required",
    "manager_role",
    "direct_reports_count",
    "posting_quality",
    "language_requirements",
    "tech_stack",
    "extraction_meta",
    "extraction_version",
)


def _normalize_html(desc: str) -> str:
    if not desc:
        return ""
    if "<" in desc and ">" in desc:
        return markdownify(html.unescape(desc), heading_style="ATX").strip()
    return desc


def _row_features(row: pd.Series) -> dict:
    desc = _normalize_html(row.get("description_md") or "")
    title = row.get("title") or ""
    feats = extract_features(desc, title=title)

    # Roll mined salary into the structured columns + USD-yearly normalization,
    # mirroring normalize.normalize().
    salary_min = row.get("salary_min")
    salary_max = row.get("salary_max")
    currency = row.get("salary_currency")
    period = row.get("salary_period")
    disclosed = bool(row.get("salary_disclosed") or False)

    if not disclosed and feats.get("salary_disclosed"):
        salary_min = feats["salary_min"]
        salary_max = feats["salary_max"]
        currency = feats["salary_currency"]
        period = feats["salary_period"]
        disclosed = True

    smy_min, smy_max = normalize_salary(salary_min, salary_max, currency, period)

    # Title cleanups (these were already done at ingest, but redo for safety).
    title_clean = title
    seniority = extract_seniority(title_clean)
    role_family = extract_role_family(title_clean)

    # Promote remote_policy_extracted when location-based was None.
    remote_policy = row.get("remote_policy")
    extracted_remote = feats.get("remote_policy_extracted")
    if (remote_policy is None or pd.isna(remote_policy)) and extracted_remote:
        country = row.get("country")
        if extracted_remote == "remote" and country in ("US", "CA"):
            remote_policy = "remote-na"
        else:
            remote_policy = extracted_remote

    out = {
        "description_md": desc,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": currency,
        "salary_period": period,
        "salary_disclosed": disclosed,
        "salary_min_usd_yearly": smy_min,
        "salary_max_usd_yearly": smy_max,
        "remote_policy": remote_policy,
        "seniority_extracted": seniority,
        "role_family_extracted": role_family,
        "extraction_meta": feats.get("extraction_meta"),
        "extraction_version": feats.get("extraction_version", "v1"),
    }
    # Carry over the rest of the cascade fields directly.
    for col in EXTRACTED_FEATURE_COLS:
        if col in out:
            continue
        if col in feats:
            out[col] = feats[col]
    return out


def backfill(input_path: Path) -> tuple[pd.DataFrame, dict]:
    started = datetime.now(timezone.utc)
    df = pd.read_parquet(input_path)
    logger.info("loaded %d rows from %s", len(df), input_path)

    updates = df.apply(_row_features, axis=1, result_type="expand")
    for col in updates.columns:
        df[col] = updates[col]

    # Compute per-feature hit rates.
    finished = datetime.now(timezone.utc)
    hit_rates: dict[str, float] = {}
    for col in EXTRACTED_FEATURE_COLS:
        if col not in df.columns:
            continue
        non_null = df[col].notna().sum()
        hit_rates[col] = round(non_null / len(df) * 100, 1)

    salary_disclosed_rate = round(df["salary_disclosed"].mean() * 100, 1)

    report = {
        "input_rows": len(df),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_sec": (finished - started).total_seconds(),
        "salary_disclosed_rate": salary_disclosed_rate,
        "feature_hit_rates": hit_rates,
    }
    return df, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/latest/jobs.parquet")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--snapshot-date", default=None, help="Override snapshot date (YYYY-MM-DD)")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    df, report = backfill(Path(args.input))

    snapshot_date = args.snapshot_date or datetime.utcnow().strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir)
    snapshot_dir = output_dir / "snapshots" / snapshot_date
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = snapshot_dir / "jobs.parquet"
    df.to_parquet(parquet_path, index=False)

    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(latest_dir / "jobs.parquet", index=False)

    feature_dir = output_dir / "reports" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / f"{snapshot_date}.json"
    feature_path.write_text(json.dumps(report, indent=2, default=str))

    logger.info("backfill done :: %d rows → %s", len(df), parquet_path)
    logger.info("salary_disclosed rate: %.1f%%", report["salary_disclosed_rate"])
    logger.info("top fill rates:")
    for col, rate in sorted(report["feature_hit_rates"].items(), key=lambda kv: -kv[1])[:15]:
        logger.info("  %-32s %5.1f%%", col, rate)

    if args.push_to_hub:
        from ingestion.push_to_hub import push_snapshot

        push_snapshot(
            snapshot_dir=snapshot_dir,
            quality_path=feature_path,  # reuse for quality slot — we wrote feature stats
            stats_path=feature_path,
            companies_yaml=Path("ingestion/companies.yaml"),
            latest_parquet=latest_dir / "jobs.parquet",
            snapshot_date=snapshot_date,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
