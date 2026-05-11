"""Publish public-facing dataset docs to the HF Dataset repo.

The dataset repo (``arjun10g/na-tech-jobs``) gets only the two md files
that are intended for end-users:

- ``README.md`` — dataset card / project overview.
- ``DATA_DICTIONARY.md`` — column-by-column reference for the curated
  parquet, with versioned-prediction columns.

Internal docs (``CLAUDE.md``, ``MAINTENANCE.md``, ``LITERATURE_REVIEW.md``)
are deliberately **not** pushed — they're contributor-facing and hosted
on the GitHub repo only.

Usage::

    uv run python -m scripts.publish_dataset_docs
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger("publish_dataset_docs")

DEFAULT_REPO = os.environ.get("HF_DATASET_REPO", "arjun10g/na-tech-jobs")

# Whitelist — anything not on this list is never pushed to the dataset repo.
ALLOWED_DOCS: tuple[str, ...] = ("README.md", "DATA_DICTIONARY.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--root", default=".", help="Project root (defaults to current dir)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be pushed without uploading",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    root = Path(args.root)

    paths = [root / name for name in ALLOWED_DOCS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            logger.warning("missing :: %s", p)
        return 1

    if args.dry_run:
        for p in paths:
            print(f"would push :: {p} → {args.repo_id}/{p.name}")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    api = HfApi(token=token)
    operations = [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in paths]
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"docs: publish {', '.join(p.name for p in paths)}",
    )
    sha = getattr(commit, "oid", "<unknown>")
    logger.info("pushed %s to %s :: %s", [p.name for p in paths], args.repo_id, sha)
    print(sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
