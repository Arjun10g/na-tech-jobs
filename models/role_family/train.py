"""Train the role-family classifier.

v1 architecture (per ``LITERATURE_REVIEW.md`` §17): frozen sentence-transformer
embeddings + multinomial logistic regression. Runs in ~30 s on CPU.

    uv run python -m models.role_family.train
    uv run python -m models.role_family.train --encoder BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import logging

from models._classifier_base import (
    DEFAULT_C_GRID,
    DEFAULT_ENCODER_ID,
    ClassifierSpec,
    train_classifier,
)

logger = logging.getLogger("models.role_family.train")

SPEC = ClassifierSpec(
    name="role_family",
    label_column="role_family_extracted",
    # Drop the "Other" regex fallback (~70% of rows — unreliable supervision).
    # Also drop "Manager" because that's already captured by `seniority` +
    # `manager_role`; role_family is about *what* the IC does (DS / MLE / DE /
    # ...), not IC-vs-people-manager. With Manager dropped, the remaining
    # 7 classes are roughly balanced at 52-151 rows each.
    drop_labels=("Other", "Manager"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", default=DEFAULT_ENCODER_ID, help="sentence-transformers model id")
    p.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_C_GRID),
        help="L2 inverse-strength grid for 5-fold CV",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    train_classifier(
        SPEC,
        encoder_id=args.encoder,
        c_grid=tuple(args.c_grid),
        cv_folds=args.cv_folds,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
