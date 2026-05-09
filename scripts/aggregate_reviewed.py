"""Aggregate reviewer agent outputs into the final ``eval/<classifier>_test.jsonl``.

After ``scripts.build_review_packets`` + 10 reviewer agents (5 per classifier)
run, each shard has a ``reviewed_NN.jsonl`` file. This script merges them,
drops `skipped` rows, and writes the gold test set with a stable
``source: claude-reviewed`` provenance flag.

Run::

    uv run python -m scripts.aggregate_reviewed
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger("aggregate_reviewed")

CLASSIFIERS = ("seniority", "role_family")


def aggregate(classifier: str) -> dict:
    src_dir = Path(f"data/eval_review_packets/{classifier}")
    out_dir = Path("eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{classifier}_test.jsonl"

    rows: list[dict] = []
    source_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    n_skipped = 0
    for path in sorted(src_dir.glob("reviewed_*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = rec.get("source", "accepted")
            source_counter[source] += 1
            if source == "skipped" or rec.get("label") is None:
                n_skipped += 1
                continue
            rows.append(
                {
                    "id": rec["id"],
                    "label": rec["label"],
                    "source": f"claude-reviewed:{source}",
                    "llm_proposal": rec.get("llm_proposal"),
                    "classifier_prediction": rec.get("classifier_prediction"),
                    "notes": rec.get("notes", ""),
                }
            )
            label_counter[rec["label"]] += 1

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "classifier": classifier,
        "n_rows": len(rows),
        "n_skipped": n_skipped,
        "by_source": dict(source_counter),
        "by_label": dict(label_counter),
        "out_path": str(out_path),
    }
    logger.info("%s :: %s", classifier, summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    summaries = []
    for cls in CLASSIFIERS:
        summaries.append(aggregate(cls))
    Path("eval/reviewed_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(json.dumps(summaries, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
