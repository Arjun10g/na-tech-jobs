"""Score every active job in the curated table with the Phase 2-4 models
and write versioned prediction columns back.

Per CLAUDE.md §6 the predictions are tagged with a model version (e.g.
``predicted_salary_usd_v1``, ``seniority_label_v1``) so older predictions
remain readable when we retrain. The wide table is the single source of
truth for downstream consumers (Phase 5 RAG payload, Phase 7 NL→SQL).

Run::

    uv run python -m curated.enrich --push-to-hub

Each model is loaded lazily; missing models are skipped with a warning so
the script can be re-run after a partial training (e.g. seniority finished
but role_family hasn't).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("curated.enrich")

PHASE_4_VERSION = "v1"


def _load_curated(curated_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(curated_path)
    logger.info("loaded %d active rows from %s", len(df), curated_path)
    return df


def _build_text(df: pd.DataFrame) -> list[str]:
    title = df.get("title", pd.Series([""] * len(df))).fillna("").astype(str)
    description = df.get("description_md", pd.Series([""] * len(df))).fillna("").astype(str)
    return (title + " — " + description.str[:1500]).tolist()


# ── Per-model scorers ─────────────────────────────────────────────────────


def _load_classifier(name: str, hub_loader, local_dir: Path):
    """Prefer local artifact (post-train, pre-push); fall back to Hub."""
    if (local_dir / "classifier.joblib").exists():
        from models._classifier_inference import LinearProbeClassifier

        logger.info("%s :: loading local artifact %s", name, local_dir)
        return LinearProbeClassifier.load(local_dir)
    return hub_loader()


def _score_seniority(df: pd.DataFrame, batch_size: int = 64) -> tuple[list[str], list[float]]:
    from models.seniority.predict import SeniorityClassifier

    try:
        clf = _load_classifier(
            "seniority",
            SeniorityClassifier.load_from_hub,
            Path("data/models/seniority/final"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("seniority model unavailable, skipping :: %s", exc)
        return ["unknown"] * len(df), [float("nan")] * len(df)

    texts = _build_text(df)
    labels = clf.predict(texts, batch_size=batch_size)
    probas = clf.predict_proba(texts, batch_size=batch_size)
    confidences = probas.max(axis=1).tolist()
    return labels, confidences


def _score_role_family(df: pd.DataFrame, batch_size: int = 64) -> tuple[list[str], list[float]]:
    from models.role_family.predict import RoleFamilyClassifier

    try:
        clf = _load_classifier(
            "role_family",
            RoleFamilyClassifier.load_from_hub,
            Path("data/models/role_family/final"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("role_family model unavailable, skipping :: %s", exc)
        return ["unknown"] * len(df), [float("nan")] * len(df)

    texts = _build_text(df)
    labels = clf.predict(texts, batch_size=batch_size)
    probas = clf.predict_proba(texts, batch_size=batch_size)
    confidences = probas.max(axis=1).tolist()
    return labels, confidences


def _score_skills(df: pd.DataFrame, batch_size: int = 8) -> list[list[str]]:
    from models.skills.predict import SkillExtractor

    extractor = SkillExtractor()
    if not extractor._ensure_loaded():
        logger.warning("skills extractor unavailable, skipping")
        return [[] for _ in range(len(df))]

    items = list(
        zip(
            df.get("description_md", pd.Series([""] * len(df))).fillna("").tolist(),
            df.get("title", pd.Series([""] * len(df))).fillna("").tolist(),
            strict=True,
        )
    )
    return extractor.extract_batch(items, batch_size=batch_size)


def _score_salary(df: pd.DataFrame) -> list[float]:
    from models.salary.predict import SalaryPredictor

    local_path = Path("data/models/salary/salary_predictor.joblib")
    try:
        if local_path.exists():
            logger.info("salary :: loading local artifact %s", local_path)
            predictor = SalaryPredictor.load(local_path)
        else:
            from app.model_loader import get_predictor

            predictor = get_predictor()
    except Exception as exc:  # noqa: BLE001
        logger.warning("salary predictor unavailable, skipping :: %s", exc)
        return [float("nan")] * len(df)

    feature_columns = [
        "min_years_experience",
        "min_education",
        "seniority_extracted",
        "manager_role",
        "clearance_level",
        "country",
        "source",
        "role_family_extracted",
        "remote_policy",
        "contract_type",
        "equity_form",
        "bonus_type",
        "region",
        "city",
        "requires_security_clearance",
        "offers_visa_sponsorship",
        "offers_relocation",
        "offers_equity",
        "bonus_mentioned",
        "on_call_required",
        "requires_citizenship",
        "language_requirements",
        "tech_stack",
        "posted_at",
    ]
    available = [c for c in feature_columns if c in df.columns]
    X = df[available].copy()  # noqa: N806 — sklearn convention
    log_preds = predictor.predict_log_usd_yearly(X)
    return [float(10.0**v) for v in log_preds]


# ── Orchestrator ──────────────────────────────────────────────────────────


def enrich(
    curated_path: Path = Path("data/curated/jobs.parquet"),
    output_dir: Path = Path("data/curated_enriched"),
    *,
    skip_skills: bool = False,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    df = _load_curated(curated_path)

    # Score salary BEFORE the sentence-transformer classifiers. Reason:
    # joblib-loading the salary pipeline after PyTorch has touched MPS
    # segfaults on macOS (joblib/threadpoolctl interacting with the MPS
    # allocator). Doing salary first avoids the conflict entirely; each
    # scorer is independent and tolerant of missing models.
    logger.info("scoring salary …")
    df[f"predicted_salary_usd_{PHASE_4_VERSION}"] = _score_salary(df)

    logger.info("scoring seniority …")
    seniority_labels, seniority_conf = _score_seniority(df)
    df[f"seniority_label_{PHASE_4_VERSION}"] = seniority_labels
    df[f"seniority_confidence_{PHASE_4_VERSION}"] = seniority_conf

    logger.info("scoring role_family …")
    role_labels, role_conf = _score_role_family(df)
    df[f"role_family_{PHASE_4_VERSION}"] = role_labels
    df[f"role_family_confidence_{PHASE_4_VERSION}"] = role_conf

    if skip_skills:
        df[f"extracted_skills_{PHASE_4_VERSION}"] = [[] for _ in range(len(df))]
    else:
        logger.info("extracting skills (NuExtract) …")
        df[f"extracted_skills_{PHASE_4_VERSION}"] = _score_skills(df)

    df["prediction_model_version"] = PHASE_4_VERSION

    # Coverage stats — what fraction of rows got a real (non-default) label.
    coverage = {
        "seniority_label": int(
            (~df[f"seniority_label_{PHASE_4_VERSION}"].isin(("unknown",))).sum()
        ),
        "role_family": int((~df[f"role_family_{PHASE_4_VERSION}"].isin(("unknown",))).sum()),
        "extracted_skills": int(
            df[f"extracted_skills_{PHASE_4_VERSION}"].apply(lambda x: bool(x)).sum()
        ),
        "predicted_salary_usd": int(df[f"predicted_salary_usd_{PHASE_4_VERSION}"].notna().sum()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "jobs.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("wrote %s :: %d rows × %d cols", out_path, *df.shape)

    finished = datetime.now(timezone.utc)
    stats = {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_sec": (finished - started).total_seconds(),
        "n_rows": int(len(df)),
        "model_version": PHASE_4_VERSION,
        "coverage": coverage,
    }
    stats_path = output_dir / "enrich_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    return stats


def push_enriched_to_hub(
    output_dir: Path,
    *,
    repo_id: str | None = None,
) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    repo_id = repo_id or os.environ.get("HF_DATASET_REPO", "arjun10g/na-tech-jobs")
    api = HfApi(token=token)
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    operations = [
        CommitOperationAdd(
            path_in_repo="curated_enriched/jobs.parquet",
            path_or_fileobj=str(output_dir / "jobs.parquet"),
        ),
        CommitOperationAdd(
            path_in_repo=f"curated_enriched/enrich_stats/{snapshot_date}.json",
            path_or_fileobj=str(output_dir / "enrich_stats.json"),
        ),
    ]
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"enrich: Phase 4 v1 predictions over {snapshot_date}",
    )
    return getattr(commit, "oid", "<unknown>")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curated-path", default="data/curated/jobs.parquet")
    p.add_argument("--output-dir", default="data/curated_enriched")
    p.add_argument("--skip-skills", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    stats = enrich(
        curated_path=Path(args.curated_path),
        output_dir=Path(args.output_dir),
        skip_skills=args.skip_skills,
    )
    print("\n=== enrichment summary ===")
    print(json.dumps(stats, indent=2, default=str))
    if args.push_to_hub:
        sha = push_enriched_to_hub(Path(args.output_dir))
        logger.info("pushed enriched dataset :: %s", sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
