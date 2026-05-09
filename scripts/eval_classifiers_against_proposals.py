"""Score the trained classifiers against the LLM-proposed eval set.

This is a **preliminary** metric — the LLM proposals are not yet
human-reviewed. After ``scripts.label_classifier --review`` we'd re-run
this against the human-verified labels. For now the LLM-vs-classifier
agreement gives us:

1. A sanity check that the classifier's regex-agreement F1 generalizes
   when measured against an independent labeler.
2. A pre-review baseline so the model card can report something more
   honest than "F1 vs regex on its own training-distribution slice."

Run::

    uv run python -m scripts.eval_classifiers_against_proposals
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

logger = logging.getLogger("eval_classifiers")

CLASSIFIERS = ("seniority", "role_family")


def _load_proposals(proposals_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(proposals_dir.glob("labels_*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _bootstrap_f1_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate(
    classifier_name: str,
    curated_path: Path = Path("data/curated/jobs.parquet"),
    proposals_dir: Path | None = None,
) -> dict:
    proposals_dir = proposals_dir or Path(f"data/eval_proposals/{classifier_name}")
    if classifier_name == "seniority":
        from models.seniority.predict import SeniorityClassifier

        clf = SeniorityClassifier.load(Path("data/models/seniority/final"))
    else:
        from models.role_family.predict import RoleFamilyClassifier

        clf = RoleFamilyClassifier.load(Path("data/models/role_family/final"))

    proposals = _load_proposals(proposals_dir)
    logger.info("loaded %d proposals from %s", len(proposals), proposals_dir)

    df = pd.read_parquet(curated_path).set_index("id")

    rows = []
    for rec in proposals.to_dict(orient="records"):
        if rec["id"] not in df.index:
            continue
        row = df.loc[rec["id"]]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        title = row.get("title") or ""
        desc = (row.get("description_md") or "")[:1000]
        text = f"{title} — {desc}"
        rows.append(
            {
                "id": rec["id"],
                "text": text,
                "llm_label": rec["llm_label"],
                "llm_confidence": rec.get("confidence", "high"),
            }
        )
    eval_df = pd.DataFrame(rows)

    # Restrict to labels the classifier actually knows about (LLM may have
    # used 9 classes whereas the classifier was trained on the regex-only
    # subset). Rows with out-of-vocab labels are reported separately.
    known_labels = set(clf.id2label.values())
    eval_df["in_vocab"] = eval_df["llm_label"].isin(known_labels)
    in_vocab = eval_df[eval_df["in_vocab"]].copy()
    out_of_vocab = eval_df[~eval_df["in_vocab"]].copy()

    preds = clf.predict(in_vocab["text"].tolist(), batch_size=64)
    in_vocab["pred"] = preds

    y_true = in_vocab["llm_label"].to_numpy()
    y_pred = in_vocab["pred"].to_numpy()
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_lo, f1_hi = _bootstrap_f1_ci(y_true, y_pred)

    # High-confidence-only subset.
    high_conf = in_vocab[in_vocab["llm_confidence"] == "high"]
    if len(high_conf) >= 20:
        hc_acc = float(accuracy_score(high_conf["llm_label"], high_conf["pred"]))
        hc_f1 = float(
            f1_score(high_conf["llm_label"], high_conf["pred"], average="macro", zero_division=0)
        )
    else:
        hc_acc = hc_f1 = float("nan")

    report = classification_report(y_true, y_pred, zero_division=0, digits=3, output_dict=True)

    summary = {
        "classifier": classifier_name,
        "n_proposals_total": len(proposals),
        "n_in_vocab": len(in_vocab),
        "n_out_of_vocab": len(out_of_vocab),
        "out_of_vocab_labels": (
            out_of_vocab["llm_label"].value_counts().to_dict() if len(out_of_vocab) else {}
        ),
        "vs_llm": {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "f1_macro_ci95": [round(f1_lo, 4), round(f1_hi, 4)],
        },
        "vs_llm_high_confidence": {
            "n": int(len(high_conf)),
            "accuracy": round(hc_acc, 4) if hc_acc == hc_acc else None,  # noqa: PLR0124
            "f1_macro": round(hc_f1, 4) if hc_f1 == hc_f1 else None,  # noqa: PLR0124
        },
        "classification_report": report,
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classifier", choices=("seniority", "role_family", "both"), default="both")
    p.add_argument("--curated-path", default="data/curated/jobs.parquet")
    p.add_argument("--out-dir", default="eval/preliminary")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets: Iterable[str] = (
        ("seniority", "role_family") if args.classifier == "both" else (args.classifier,)
    )

    for cls in targets:
        logger.info("=== %s ===", cls)
        summary = evaluate(cls, curated_path=Path(args.curated_path))
        out_path = out_dir / f"{cls}_vs_llm.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(
            "%s :: vs LLM accuracy=%.3f f1_macro=%.3f (CI [%s])  high-conf-f1_macro=%s",
            cls,
            summary["vs_llm"]["accuracy"],
            summary["vs_llm"]["f1_macro"],
            ", ".join(str(x) for x in summary["vs_llm"]["f1_macro_ci95"]),
            summary["vs_llm_high_confidence"]["f1_macro"],
        )
        logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
