"""Deterministic train / test split for the salary regressor.

We freeze the test-set membership **before** any modelling decision so the
audit findings can't subtly steer feature engineering toward the test
distribution. The split is stratified by ``country × source`` so US/CA
proportions and Greenhouse/Lever/Ashby proportions match across folds.

Because the split is keyed off the canonical ``id`` (sha256-based) and a
fixed ``seed``, it's reproducible across machines without persisting the
parquet itself. The committed ``data/eda/test_split_ids.json`` is a
convenience artifact for cross-checking and for downstream notebooks
that don't want to recompute the hash.

Usage::

    from eda.split import freeze_split, load_test_ids
    splits = freeze_split(curated_df, test_frac=0.2, seed=42)
    train_df = curated_df[curated_df.id.isin(splits["train_ids"])]
    test_df  = curated_df[curated_df.id.isin(splits["test_ids"])]

    uv run python -m eda.split   # writes data/eda/test_split_ids.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("eda.split")

DEFAULT_TEST_FRAC: float = 0.20
DEFAULT_SEED: int = 42
STRATIFY_KEYS: tuple[str, ...] = ("country", "source")


def _hash_to_unit(job_id: str, seed: int) -> float:
    """Map ``id`` + seed deterministically into [0, 1)."""
    digest = hashlib.sha256(f"{seed}::{job_id}".encode()).hexdigest()
    return int(digest[:16], 16) / 0xFFFFFFFFFFFFFFFF


def freeze_split(
    df: pd.DataFrame,
    *,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_SEED,
    stratify_by: tuple[str, ...] = STRATIFY_KEYS,
) -> dict[str, Any]:
    """Return ``{train_ids, test_ids, stratum_counts, params}``.

    Stratification: within each unique combination of ``stratify_by`` values,
    rows whose hashed unit score < ``test_frac`` go to test. Hash is keyed by
    ``seed`` + ``id`` so two re-runs with the same seed give identical splits.
    """
    if "id" not in df.columns:
        raise KeyError("df must have an 'id' column")
    df = df.copy()
    df["__h"] = df["id"].astype(str).apply(lambda x: _hash_to_unit(x, seed))
    df["__test"] = df["__h"] < test_frac

    stratum_counts: list[dict[str, Any]] = []
    for stratum, grp in df.groupby(list(stratify_by), dropna=False):
        stratum_counts.append(
            {
                **dict(
                    zip(
                        stratify_by,
                        stratum if isinstance(stratum, tuple) else (stratum,),
                        strict=False,
                    )
                ),
                "n": int(len(grp)),
                "n_test": int(grp["__test"].sum()),
                "n_train": int((~grp["__test"]).sum()),
                "test_frac_actual": round(float(grp["__test"].mean()), 4),
            }
        )

    train_ids = df.loc[~df["__test"], "id"].astype(str).tolist()
    test_ids = df.loc[df["__test"], "id"].astype(str).tolist()
    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "stratum_counts": stratum_counts,
        "params": {
            "test_frac": test_frac,
            "seed": seed,
            "stratify_by": list(stratify_by),
            "n_total": int(len(df)),
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "test_frac_actual": round(len(test_ids) / max(len(df), 1), 4),
        },
    }


def write_split(splits: dict[str, Any], out_path: Path) -> None:
    """Persist a hash-only manifest (no row data) to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=2, default=str))


def load_test_ids(manifest_path: Path) -> set[str]:
    """Read the persisted manifest and return the test-id set."""
    obj = json.loads(manifest_path.read_text())
    return set(obj["test_ids"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/curated/jobs.parquet")
    p.add_argument("--output", default="data/eda/test_split_ids.json")
    p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    df = pd.read_parquet(args.input)
    logger.info("loaded %d rows from %s", len(df), args.input)

    splits = freeze_split(df, test_frac=args.test_frac, seed=args.seed)
    out_path = Path(args.output)
    write_split(splits, out_path)
    logger.info(
        "wrote %s :: train=%d, test=%d, test_frac_actual=%.3f",
        out_path,
        splits["params"]["n_train"],
        splits["params"]["n_test"],
        splits["params"]["test_frac_actual"],
    )
    for s in splits["stratum_counts"]:
        logger.info(
            "  stratum %s :: n=%d  test=%d  test_frac=%.3f",
            {k: s[k] for k in STRATIFY_KEYS},
            s["n"],
            s["n_test"],
            s["test_frac_actual"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
