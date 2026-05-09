"""CLI for hand-labelling a clean test set for the title classifiers.

Per CLAUDE.md §7 the v1 metrics are weak-supervised (regex agreement).
Phase 4+ wants a hand-labelled test set of ~500 examples to report
unbiased F1 / per-class confusion. This CLI samples diverse titles +
descriptions from the curated parquet and lets you assign labels
interactively, saving to ``eval/<classifier>_test.jsonl``.

Run::

    uv run python -m scripts.label_classifier seniority --n 50
    uv run python -m scripts.label_classifier role_family --n 100

Each label is appended; relaunch to resume. Stratified sampling pulls
from titles whose regex DIDN'T fire (i.e. the rows the model is
expected to generalize to) plus a few from each existing class for
coverage.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("label_classifier")

LABELS_BY_CLASSIFIER: dict[str, list[str]] = {
    "seniority": [
        "intern",
        "junior",
        "mid",
        "senior",
        "staff",
        "principal",
        "manager",
        "director",
        "exec",
    ],
    "role_family": [
        "DS",
        "DA",
        "DE",
        "MLE",
        "RS",
        "AS",
        "SWE-ML",
        "Manager",
        "Other",
    ],
}


def _existing_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen: set[str] = set()
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            seen.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return seen


def _sample_titles(
    df: pd.DataFrame,
    classifier: str,
    n: int,
    seen: set[str],
) -> pd.DataFrame:
    label_col = f"{classifier}_extracted" if classifier == "seniority" else "role_family_extracted"
    fallbacks = {"seniority": "mid", "role_family": "Other"}
    fallback = fallbacks.get(classifier)

    df = df[~df["id"].isin(seen)]
    target_each = max(1, n // (len(LABELS_BY_CLASSIFIER[classifier]) + 1))

    chunks: list[pd.DataFrame] = []
    # Half the budget on regex fallback rows (the model's actual job).
    if fallback and (df[label_col] == fallback).any():
        chunks.append(
            df[df[label_col] == fallback].sample(
                n=min(n // 2, (df[label_col] == fallback).sum()),
                random_state=42,
            )
        )
    # Stratified sample on the explicit-match labels for coverage.
    for label in LABELS_BY_CLASSIFIER[classifier]:
        if label == fallback:
            continue
        sub = df[df[label_col] == label]
        if sub.empty:
            continue
        chunks.append(sub.sample(n=min(target_each, len(sub)), random_state=42))
    sample = pd.concat(chunks).drop_duplicates(subset=["id"]).head(n)
    return sample


def _print_row(row: pd.Series, classifier: str) -> None:
    print("\n" + "═" * 90)
    print(f"  ID: {row['id']}")
    print(f"  Company: {row.get('company_name')}")
    print(f"  Title: {row.get('title')}")
    print(f"  Location: {row.get('location_raw')}")
    label_col = f"{classifier}_extracted" if classifier == "seniority" else "role_family_extracted"
    print(f"  Regex label: {row.get(label_col)}")
    desc = (row.get("description_md") or "")[:600]
    print(f"\n  Description excerpt:\n  {desc[:560]}…")
    print("─" * 90)


def _aggregate_proposals(classifier: str, proposals_dir: Path) -> dict[str, dict]:
    """Read all `labels_*.jsonl` files written by the LLM-propose agents."""
    by_id: dict[str, dict] = {}
    for path in sorted(proposals_dir.glob("labels_*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in rec and "llm_label" in rec:
                by_id[rec["id"]] = rec
    return by_id


def _print_review_row(row: pd.Series, classifier: str, proposal: dict) -> None:
    print("\n" + "═" * 90)
    print(f"  ID: {row['id']}")
    print(f"  Company: {row.get('company_name')}")
    print(f"  Title: {row.get('title')}")
    print(f"  Location: {row.get('location_raw')}")
    label_col = f"{classifier}_extracted" if classifier == "seniority" else "role_family_extracted"
    print(f"  Regex label : {row.get(label_col)}")
    confidence = proposal.get("confidence", "?")
    print(f"  LLM proposal: {proposal['llm_label']:>10}   ({confidence} confidence)")
    desc = (row.get("description_md") or "")[:600]
    print(f"\n  Description excerpt:\n  {desc[:560]}…")
    print("─" * 90)


def _review_loop(
    classifier: str,
    df: pd.DataFrame,
    proposals_dir: Path,
    out_path: Path,
) -> int:
    proposals = _aggregate_proposals(classifier, proposals_dir)
    seen = _existing_ids(out_path)
    todo_ids = [pid for pid in proposals if pid not in seen]
    if not todo_ids:
        print(f"Nothing to review — all {len(proposals)} proposals already labeled in {out_path}.")
        return 0

    df_indexed = df.set_index("id")
    valid_labels = LABELS_BY_CLASSIFIER[classifier]
    label_help = " | ".join(f"({i}) {lbl}" for i, lbl in enumerate(valid_labels))

    print(f"\nReviewing {len(todo_ids)} proposed labels for {classifier}.")
    print(f"Source: {proposals_dir}.")
    print(f"Labels: {label_help}")
    print("Press ENTER to ACCEPT the LLM proposal, type a label/number to OVERRIDE,")
    print("'?' to skip, 's' to stop.\n")

    n_accepted = n_overridden = n_skipped = 0
    with out_path.open("a") as f:
        for pid in todo_ids:
            if pid not in df_indexed.index:
                continue
            row = df_indexed.loc[pid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            row = pd.Series(row).rename({"index": "id"})
            row["id"] = pid
            proposal = proposals[pid]

            _print_review_row(row, classifier, proposal)
            while True:
                resp = input(f"\n[ENTER=accept '{proposal['llm_label']}'] override? > ").strip()
                if resp == "":
                    label = proposal["llm_label"]
                    source = "llm-accepted"
                    n_accepted += 1
                    break
                if resp.lower() == "s":
                    print(
                        f"\nStopped. Accepted={n_accepted} Overridden={n_overridden} "
                        f"Skipped={n_skipped}. Saved to {out_path}."
                    )
                    return 0
                if resp == "?":
                    n_skipped += 1
                    label = None
                    source = None
                    break
                if resp.isdigit() and 0 <= int(resp) < len(valid_labels):
                    label = valid_labels[int(resp)]
                    source = "human-override"
                    n_overridden += 1
                    break
                if resp in valid_labels:
                    label = resp
                    source = "human-override"
                    n_overridden += 1
                    break
                print(f"  invalid; choose from {valid_labels}, ENTER, ?, or s")

            if label is None:
                continue
            f.write(
                json.dumps(
                    {
                        "id": pid,
                        "title": row.get("title"),
                        "label": label,
                        "llm_proposal": proposal["llm_label"],
                        "llm_confidence": proposal.get("confidence"),
                        "source": source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()

    print(
        f"\nReview complete. Accepted={n_accepted} Overridden={n_overridden} "
        f"Skipped={n_skipped}. Saved to {out_path}."
    )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("classifier", choices=tuple(LABELS_BY_CLASSIFIER))
    p.add_argument(
        "--n", type=int, default=50, help="Number of titles to label this session (no-review mode)"
    )
    p.add_argument(
        "--review",
        action="store_true",
        help=(
            "Spot-check LLM proposals from data/eval_proposals/<classifier>/. "
            "Press ENTER to accept, type a label to override."
        ),
    )
    p.add_argument(
        "--proposals-dir",
        default=None,
        help="Defaults to data/eval_proposals/<classifier>",
    )
    p.add_argument("--curated-path", default="data/curated/jobs.parquet")
    p.add_argument("--out-path", default=None, help="Defaults to eval/<classifier>_test.jsonl")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    classifier = args.classifier
    out_path = Path(args.out_path or f"eval/{classifier}_test.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.curated_path)
    seen = _existing_ids(out_path)
    logger.info("loaded %d rows; %d already labelled", len(df), len(seen))

    if args.review:
        proposals_dir = Path(args.proposals_dir or f"data/eval_proposals/{classifier}")
        return _review_loop(classifier, df, proposals_dir, out_path)

    sample = _sample_titles(df, classifier, args.n, seen)
    if sample.empty:
        print("Nothing left to label.")
        return 0

    valid_labels = LABELS_BY_CLASSIFIER[classifier]
    label_help = " | ".join(f"({i}) {lbl}" for i, lbl in enumerate(valid_labels))
    print(f"\nLabels: {label_help}")
    print("Type the number, the label, '?' to skip, 's' to stop.\n")

    n_labelled = 0
    with out_path.open("a") as f:
        for _, row in sample.iterrows():
            _print_row(row, classifier)
            while True:
                resp = input(f"\nLabel ({label_help}, ?, s)> ").strip()
                if resp == "":
                    continue
                if resp.lower() == "s":
                    print(f"\nStopped after {n_labelled} labels. Saved to {out_path}.")
                    return 0
                if resp == "?":
                    break
                if resp.isdigit() and 0 <= int(resp) < len(valid_labels):
                    label = valid_labels[int(resp)]
                    break
                if resp in valid_labels:
                    label = resp
                    break
                print(f"  invalid; choose from {valid_labels} or ? or s")
            else:  # noqa: PLW0120
                continue
            if resp == "?":
                continue
            f.write(
                json.dumps(
                    {"id": row["id"], "title": row.get("title"), "label": label, "source": "human"},
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            n_labelled += 1

    print(f"\nLabelled {n_labelled} rows. Saved to {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
