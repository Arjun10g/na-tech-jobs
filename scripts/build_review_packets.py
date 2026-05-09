"""Combine proposals + classifier predictions + shard data into review packets.

For each shard at ``data/eval_proposals/<classifier>/shard_NN.jsonl``, look up:
- the LLM proposal (label + confidence) from ``labels_NN.jsonl``
- the trained classifier's prediction (label + confidence)

Write the merged packet to ``data/eval_review_packets/<classifier>/packet_NN.jsonl``,
ready for a reviewer agent to ingest.

Run::

    uv run python -m scripts.build_review_packets
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("build_review_packets")

CLASSIFIERS = ("seniority", "role_family")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _classifier_predictions(classifier: str, ids: list[str], curated_path: Path) -> dict[str, dict]:
    if classifier == "seniority":
        from models.seniority.predict import SeniorityClassifier

        clf = SeniorityClassifier.load(Path("data/models/seniority/final"))
    else:
        from models.role_family.predict import RoleFamilyClassifier

        clf = RoleFamilyClassifier.load(Path("data/models/role_family/final"))

    df = pd.read_parquet(curated_path).set_index("id")
    df = df.loc[df.index.isin(ids)]

    texts = []
    ordered_ids = []
    for jid in ids:
        if jid not in df.index:
            continue
        row = df.loc[jid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        title = row.get("title") or ""
        desc = (row.get("description_md") or "")[:1000]
        texts.append(f"{title} — {desc}")
        ordered_ids.append(jid)

    labels = clf.predict(texts, batch_size=64)
    probas = clf.predict_proba(texts, batch_size=64)
    return {
        jid: {"label": lbl, "confidence": float(p.max())}
        for jid, lbl, p in zip(ordered_ids, labels, probas, strict=True)
    }


def build_packets(classifier: str, *, curated_path: Path) -> Path:
    proposals_dir = Path(f"data/eval_proposals/{classifier}")
    packets_dir = Path(f"data/eval_review_packets/{classifier}")
    packets_dir.mkdir(parents=True, exist_ok=True)

    # Collect all ids across shards for one classifier-prediction batch.
    all_shards: list[tuple[Path, list[dict], list[dict]]] = []
    all_ids: list[str] = []
    for shard_path in sorted(proposals_dir.glob("shard_*.jsonl")):
        shard_idx = shard_path.stem.split("_")[1]
        labels_path = proposals_dir / f"labels_{shard_idx}.jsonl"
        shard_rows = _load_jsonl(shard_path)
        label_rows = _load_jsonl(labels_path)
        all_shards.append((shard_path, shard_rows, label_rows))
        all_ids.extend(r["id"] for r in shard_rows)

    logger.info("%s :: %d ids across %d shards", classifier, len(all_ids), len(all_shards))
    preds = _classifier_predictions(classifier, all_ids, curated_path)
    logger.info("%s :: classifier scored %d rows", classifier, len(preds))

    written: list[Path] = []
    for shard_path, shard_rows, label_rows in all_shards:
        shard_idx = shard_path.stem.split("_")[1]
        labels_by_id = {r["id"]: r for r in label_rows}
        packet_path = packets_dir / f"packet_{shard_idx}.jsonl"
        with packet_path.open("w") as f:
            for row in shard_rows:
                jid = row["id"]
                proposal = labels_by_id.get(jid, {})
                pred = preds.get(jid, {})
                packet = {
                    "id": jid,
                    "title": row.get("title"),
                    "company_name": row.get("company_name"),
                    "location_raw": row.get("location_raw"),
                    "description_md": row.get("description_md"),
                    "regex_label": row.get("regex_label"),
                    "llm_proposal": proposal.get("llm_label"),
                    "llm_confidence": proposal.get("confidence"),
                    "classifier_prediction": pred.get("label"),
                    "classifier_confidence": pred.get("confidence"),
                }
                f.write(json.dumps(packet, ensure_ascii=False) + "\n")
        written.append(packet_path)
        logger.info("wrote %s (%d rows)", packet_path, len(shard_rows))
    return packets_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curated-path", default="data/curated/jobs.parquet")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    for cls in CLASSIFIERS:
        build_packets(cls, curated_path=Path(args.curated_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
