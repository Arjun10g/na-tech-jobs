"""Drift detection between two snapshots of the curated parquet.

Per CLAUDE.md §8 the weekly drift cron compares the latest snapshot
against a 4-week rolling baseline. This module is the runner — given a
``current`` and a ``reference`` parquet path, it produces:

- An Evidently HTML report (saved to ``reports/drift/<date>.html``)
- A ``metrics.json`` with the key numerics — PSI per feature, drift
  flags per column — that the dashboard tab and the alerter consume
  without parsing HTML.

Threshold: PSI > 0.20 on any tracked feature OR drift_share > 0.30
across all features → flag the next monthly retrain as ``priority``.
We log the breach via Discord (``monitoring/alerts.py``) and persist a
``priority=true`` marker in ``reports/drift/<date>/priority.json`` so the
retrain workflow can pick it up.

For the v1 demo (only one snapshot exists), this module also exposes
``synthetic_split()`` which random-shuffles the curated parquet into two
halves so the dashboard has *something* to render. Real drift only
activates once two real snapshots are available.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("monitoring.drift")

# Features watched for drift. Anything not on this list still appears in
# the Evidently report but doesn't gate retraining priority.
TRACKED_NUMERICAL: tuple[str, ...] = (
    "salary_min_usd_yearly",
    "salary_max_usd_yearly",
    "predicted_salary_usd_v1",
    "min_years_experience",
    "seniority_confidence_v1",
    "role_family_confidence_v1",
)
TRACKED_CATEGORICAL: tuple[str, ...] = (
    "country",
    "remote_policy",
    "source",
    "seniority_extracted",
    "role_family_extracted",
    "seniority_label_v1",
    "role_family_v1",
    "salary_disclosed",
)

# Threshold flags surfaced to the dashboard / alerter / retrain trigger.
PSI_THRESHOLD: float = 0.20
DRIFT_SHARE_THRESHOLD: float = 0.30


@dataclass
class DriftResult:
    """The slimmer-than-Evidently view we hand to downstream consumers."""

    current_path: str
    reference_path: str
    n_current: int
    n_reference: int
    columns_drifted: list[str]
    columns_total: int
    drift_share: float
    priority_breach: bool  # any tracked feature crossed PSI_THRESHOLD
    breached_features: list[str]
    html_path: str | None
    metrics_path: str
    generated_at: str


# ── Helpers ───────────────────────────────────────────────────────────────


def _select_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Keep only tracked columns that actually exist on this parquet."""
    nums = [c for c in TRACKED_NUMERICAL if c in df.columns]
    cats = [c for c in TRACKED_CATEGORICAL if c in df.columns]
    return nums, cats


def _coerce_for_evidently(
    df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> pd.DataFrame:
    """Evidently is pickier than pandas: numpy arrays in cells (e.g.
    ``extracted_skills_v1``) need to be dropped, NaN floats stay, and
    bools stringify cleanly. Returns a DataFrame restricted to the
    tracked columns with consistent dtypes.
    """
    keep = num_cols + cat_cols
    out = df[keep].copy()
    # Cast categoricals to string (Evidently treats numeric-looking
    # categoricals as numerical otherwise).
    for c in cat_cols:
        out[c] = out[c].astype("string")
    return out


# ── Synthetic split (v1 demo) ─────────────────────────────────────────────


def synthetic_split(
    parquet_path: Path,
    *,
    seed: int = 42,
    perturb_role_family: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one parquet into reference / current halves.

    For v1 demo only — produces two non-overlapping halves with optional
    light perturbation on the "current" half so drift > 0 is visible.
    Real production drift compares two real snapshots.
    """
    df = pd.read_parquet(parquet_path)
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    half = len(shuffled) // 2
    reference = shuffled.iloc[:half].copy()
    current = shuffled.iloc[half:].copy()

    if perturb_role_family and "role_family_v1" in current.columns:
        # Bias the current half toward MLE / RS to make drift visible.
        rng = pd.Series(range(len(current))).sample(frac=0.05, random_state=seed)
        current.loc[current.index[rng.values], "role_family_v1"] = "MLE"
    return reference, current


# ── Drift report ──────────────────────────────────────────────────────────


def run_drift_report(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    out_dir: Path,
    *,
    snapshot_date: str | None = None,
) -> DriftResult:
    """Run Evidently's data-drift preset and emit HTML + metrics.json.

    Both DataFrames must be the curated schema. We restrict to tracked
    columns before passing to Evidently so the HTML doesn't bloat with
    free-text fields like description_md.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    snapshot_date = snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    num_cols, cat_cols = _select_columns(current)
    if not num_cols and not cat_cols:
        raise ValueError("no tracked columns found on the current parquet")

    cur = _coerce_for_evidently(current, num_cols, cat_cols)
    ref = _coerce_for_evidently(reference, num_cols, cat_cols)

    schema = DataDefinition(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
    )
    cur_ds = Dataset.from_pandas(cur, data_definition=schema)
    ref_ds = Dataset.from_pandas(ref, data_definition=schema)

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{snapshot_date}.html"
    snapshot.save_html(str(html_path))

    # Pull the slim numbers out of the snapshot for metrics.json.
    summary = snapshot.dict()
    columns_total = len(num_cols) + len(cat_cols)
    columns_drifted: list[str] = []
    psi_per_col: dict[str, float] = {}
    for m in summary.get("metrics", []):
        # Evidently's DataDriftPreset emits per-column drift metrics. We
        # extract column names + PSI / wasserstein / chi-square scores
        # whichever was used.
        if not isinstance(m, dict):
            continue
        col = m.get("column_name") or m.get("column")
        score = m.get("drift_score")
        drifted = m.get("drift_detected")
        if col and score is not None:
            psi_per_col[col] = float(score)
        if col and drifted:
            columns_drifted.append(col)

    drift_share = len(columns_drifted) / columns_total if columns_total else 0.0
    breached = [
        c
        for c, s in psi_per_col.items()
        if c in (set(TRACKED_NUMERICAL) | set(TRACKED_CATEGORICAL)) and s >= PSI_THRESHOLD
    ]
    priority_breach = bool(breached) or drift_share >= DRIFT_SHARE_THRESHOLD

    metrics = {
        "snapshot_date": snapshot_date,
        "n_current": int(len(current)),
        "n_reference": int(len(reference)),
        "columns_total": columns_total,
        "columns_drifted": columns_drifted,
        "columns_drifted_count": len(columns_drifted),
        "drift_share": round(drift_share, 4),
        "psi_per_column": {k: round(v, 4) for k, v in psi_per_col.items()},
        "priority_breach": priority_breach,
        "breached_features": breached,
        "psi_threshold": PSI_THRESHOLD,
        "drift_share_threshold": DRIFT_SHARE_THRESHOLD,
    }
    metrics_path = out_dir / f"{snapshot_date}.metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

    if priority_breach:
        priority_path = out_dir / f"{snapshot_date}.priority.json"
        priority_path.write_text(
            json.dumps(
                {
                    "snapshot_date": snapshot_date,
                    "reason": "PSI threshold breach" if breached else "drift_share threshold",
                    "breached_features": breached,
                    "drift_share": metrics["drift_share"],
                },
                indent=2,
                default=str,
            )
        )
        logger.warning(
            "DRIFT priority breach :: %d feature(s): %s (drift_share=%.2f)",
            len(breached),
            breached,
            drift_share,
        )

    return DriftResult(
        current_path="(in-memory)",
        reference_path="(in-memory)",
        n_current=len(current),
        n_reference=len(reference),
        columns_drifted=columns_drifted,
        columns_total=columns_total,
        drift_share=drift_share,
        priority_breach=priority_breach,
        breached_features=breached,
        html_path=str(html_path),
        metrics_path=str(metrics_path),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--current",
        required=False,
        default=None,
        help="Current parquet path (defaults to data/curated_enriched/jobs.parquet)",
    )
    p.add_argument(
        "--reference", required=False, default=None, help="Reference parquet path (older snapshot)"
    )
    p.add_argument(
        "--synthetic-split",
        action="store_true",
        help="Demo mode: random-shuffle one parquet into ref+current halves",
    )
    p.add_argument("--out-dir", default="reports/drift")
    p.add_argument("--snapshot-date", default=None)
    p.add_argument("--alert", action="store_true", help="Send Discord alert on priority breach")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    out_dir = Path(args.out_dir)
    enriched = Path("data/curated_enriched/jobs.parquet")
    base = Path("data/curated/jobs.parquet")

    if args.synthetic_split:
        path = Path(args.current) if args.current else (enriched if enriched.exists() else base)
        ref, cur = synthetic_split(path)
        result = run_drift_report(cur, ref, out_dir, snapshot_date=args.snapshot_date)
    else:
        cur_path = Path(args.current) if args.current else (enriched if enriched.exists() else base)
        if not args.reference:
            raise SystemExit("must pass --reference <path> (or use --synthetic-split for v1 demo)")
        ref_path = Path(args.reference)
        result = run_drift_report(
            pd.read_parquet(cur_path),
            pd.read_parquet(ref_path),
            out_dir,
            snapshot_date=args.snapshot_date,
        )
        result.current_path = str(cur_path)
        result.reference_path = str(ref_path)

    print(
        json.dumps(
            {
                "drift_share": result.drift_share,
                "columns_drifted": result.columns_drifted,
                "priority_breach": result.priority_breach,
                "breached_features": result.breached_features,
                "html": result.html_path,
                "metrics": result.metrics_path,
            },
            indent=2,
        )
    )

    if args.alert and result.priority_breach:
        try:
            from monitoring.alerts import DISCORD_RED, discord_alert

            discord_alert(
                title="🚨 Drift priority breach",
                description=(
                    f"{len(result.breached_features)} feature(s) over PSI={PSI_THRESHOLD} "
                    f"or drift_share={result.drift_share:.2f}"
                ),
                color=DISCORD_RED,
                fields={
                    "Breached features": ", ".join(result.breached_features)
                    or "(global drift_share)",
                    "Snapshot": result.generated_at,
                    "HTML": result.html_path or "—",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Discord alert failed :: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
