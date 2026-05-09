"""Push the local Qdrant index tarball to the HF Dataset repo.

The matcher tab on the live Space needs a populated ``data/qdrant/``
directory. The Space's deploy workflow excludes ``data/`` (parquets are
big, snapshots aren't reproducible), so we ship the index out-of-band
via the dataset repo:

    qdrant/qdrant_<encoder>_<version>.tar.gz

``app/retriever_loader.py`` pulls + extracts on first request.

Usage::

    # Tarball the index
    tar -cf - -C data qdrant | gzip -9 > data/qdrant_archives/qdrant_minilm_v1.tar.gz

    # Push (this script)
    uv run python -m scripts.publish_qdrant_index \\
      --tarball data/qdrant_archives/qdrant_minilm_v1.tar.gz \\
      --remote-name qdrant_minilm_v1.tar.gz
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger("publish_qdrant_index")

DEFAULT_REPO = os.environ.get("HF_DATASET_REPO", "arjun10g/na-tech-jobs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tarball", required=True, help="Local tarball path")
    p.add_argument(
        "--remote-name",
        default=None,
        help="Filename in the dataset repo (defaults to the tarball's basename)",
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    tarball = Path(args.tarball)
    if not tarball.exists():
        raise FileNotFoundError(f"missing {tarball}")
    remote = args.remote_name or tarball.name
    size_mb = tarball.stat().st_size / 1e6
    logger.info("uploading %s (%.1f MB) → %s/qdrant/%s", tarball, size_mb, args.repo_id, remote)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(tarball),
        path_in_repo=f"qdrant/{remote}",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"qdrant: {remote} ({size_mb:.0f} MB)",
    )
    logger.info("done")
    print(f"https://huggingface.co/datasets/{args.repo_id}/blob/main/qdrant/{remote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
