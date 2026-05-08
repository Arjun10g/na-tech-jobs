"""Push the winning salary regressor + model card to the HF Model repo.

Runs after ``models.salary.train`` has produced
``data/models/salary/{ladder_report.json, salary_predictor.joblib}`` and
``MODEL_CARD.md``.

::

    uv run python -m scripts.publish_salary_model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("publish_salary_model")

DEFAULT_REPO_ID = "arjun10g/na-tech-jobs-salary-v1"
DEFAULT_ARTIFACTS_DIR = Path("data/models/salary")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    p.add_argument(
        "--create", action="store_true", help="Create the model repo if it doesn't exist"
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    from huggingface_hub import CommitOperationAdd, HfApi, create_repo

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    artifacts_dir = Path(args.artifacts_dir)

    if args.create:
        try:
            create_repo(args.repo_id, repo_type="model", token=token, exist_ok=True)
            logger.info("ensured model repo %s", args.repo_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("create_repo: %s", exc)

    api = HfApi(token=token)

    operations = []
    files = [
        ("README.md", artifacts_dir / "MODEL_CARD.md"),
        ("ladder_report.json", artifacts_dir / "ladder_report.json"),
        ("leaderboard.csv", artifacts_dir / "leaderboard.csv"),
        ("salary_predictor.joblib", artifacts_dir / "salary_predictor.joblib"),
    ]
    # Per-tier coefficient / importance artifacts
    for name in (
        "tier2_mincer_coefficients.csv",
        "tier3_ridge_top_coefs.csv",
        "tier4_rf_importance.csv",
        "tier5_xgb_importance.csv",
        "tier5_xgb_best_params.json",
    ):
        files.append((f"artifacts/{name}", artifacts_dir / name))

    for path_in_repo, local_path in files:
        if not local_path.exists():
            logger.warning("missing %s; skipping", local_path)
            continue
        operations.append(
            CommitOperationAdd(
                path_in_repo=path_in_repo,
                path_or_fileobj=str(local_path),
            )
        )

    report_path = artifacts_dir / "ladder_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        commit_msg = (
            f"v1 :: winning tier = {report['winning_tier']}, "
            f"MAE ≈ {next(t for t in report['tiers'] if t['tier'] == report['winning_tier'])['mae']:.0f} USD/yr"
        )
    else:
        commit_msg = "v1 :: initial publish"

    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_msg,
    )
    sha = getattr(commit, "oid", "<unknown>")
    logger.info("pushed %d files to %s :: %s", len(operations), args.repo_id, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
