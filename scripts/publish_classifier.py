"""Publish a v1 classifier (seniority / role_family) to HF Hub.

Run AFTER ``models.<classifier>.train`` produces ``data/models/<name>/final/``::

    uv run python -m scripts.publish_classifier seniority --create
    uv run python -m scripts.publish_classifier role_family --create
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("publish_classifier")


def _model_card(name: str, summary: dict) -> str:
    eval_metrics = summary.get("eval", {})
    cv = summary.get("cv", {})
    label_counts = summary.get("label_counts", {})
    classes = ", ".join(sorted(summary.get("label2id", {}).keys()))
    encoder = summary.get("encoder_id", "sentence-transformers/all-MiniLM-L6-v2")
    n_train = summary.get("n_train", 0)
    n_val = summary.get("n_val", 0)
    n_classes = summary.get("n_classes", len(summary.get("label2id", {})))
    f1_macro = eval_metrics.get("eval_f1_macro", "n/a")
    f1_ci = eval_metrics.get("eval_f1_macro_ci95", ["n/a", "n/a"])
    accuracy = eval_metrics.get("eval_accuracy", "n/a")
    f1_weighted = eval_metrics.get("eval_f1_weighted", "n/a")
    best_c = cv.get("best_C", "n/a")
    cv_f1 = cv.get("best_f1_macro", "n/a")
    pretty_name = name.replace("_", " ")

    # Preliminary independent-labeler eval (LLM-proposed, pre-human-review).
    prelim_path = Path(f"eval/preliminary/{name}_vs_llm.json")
    prelim_block = ""
    if prelim_path.exists():
        prelim = json.loads(prelim_path.read_text())
        vs_llm = prelim["vs_llm"]
        hc = prelim["vs_llm_high_confidence"]
        prelim_block = f"""

## Independent-labeler eval (preliminary, pre-human-review)

To check the classifier didn't just memorize the regex, we sampled 230
diverse rows and had Claude (an independent labeler) propose labels in
parallel. The classifier was scored against those proposals on the
subset of rows where Claude's label is one the classifier was trained
to predict (`{prelim["n_in_vocab"]}/{prelim["n_proposals_total"]}` rows
— the rest were the regex-default labels we drop from training,
mostly `{list(prelim["out_of_vocab_labels"].keys())[0] if prelim["out_of_vocab_labels"] else "n/a"}`).

| Metric | All in-vocab proposals | LLM high-confidence subset |
|---|---|---|
| n | {prelim["n_in_vocab"]} | {hc["n"]} |
| accuracy | {vs_llm["accuracy"]} | {hc["accuracy"]} |
| f1_macro | **{vs_llm["f1_macro"]}** (95% CI [{vs_llm["f1_macro_ci95"][0]}, {vs_llm["f1_macro_ci95"][1]}]) | **{hc["f1_macro"]}** |

These numbers come from a *different labeler* than the training data, so
they're a stronger signal of generalization than the regex-agreement
metric above. Caveats: Claude's labels are themselves not gold, and the
in-vocab filter excludes the regex-default rows. A hand-reviewed gold set
is the v1.1 task — the LLM-proposed labels are the starting point for that
review (`scripts/label_classifier --review`)."""

    return f"""\
---
license: mit
library_name: scikit-learn
tags:
- text-classification
- {name}
- north-america-tech-hiring
- linear-probe
- weakly-supervised
base_model: {encoder}
metrics:
- accuracy
- f1
---

# na-tech-jobs {pretty_name} classifier — v1

A {n_classes}-class text classifier that predicts the
**{pretty_name}** of a North American tech job posting from
`title + description_md`.

## Architecture

**Frozen sentence-transformer embeddings + multinomial logistic regression.**

| Component | Choice |
|---|---|
| Encoder (frozen) | `{encoder}` |
| Pooling | mean, L2-normalized |
| Classifier | sklearn `LogisticRegression` (multinomial, lbfgs, L2) |
| Class weights | `balanced` |
| C selection | 5-fold stratified CV on f1_macro, grid `{cv.get("c_grid", [])}` |
| Selected C | `{best_c}` |

Why not full fine-tuning of DeBERTa-v3 + LoRA (the original CLAUDE.md §7
choice)? See the project's
[`LITERATURE_REVIEW.md` §17](https://github.com/Arjun10g/na-tech-jobs/blob/main/LITERATURE_REVIEW.md):
for short-text small-vocabulary classification with weakly supervised
labels (Peters et al 2019; Tunstall et al 2022, SetFit) a linear probe on
a strong general-purpose embedder reaches the same operating point at
~100x less compute. v2 will revisit fine-tuning on a hand-labeled set.

## Headline metrics (held-out 10% stratified validation)

| Metric | Value |
|---|---|
| f1_macro | **{f1_macro}** (95% CI [{f1_ci[0]}, {f1_ci[1]}]) |
| f1_weighted | {f1_weighted} |
| accuracy | {accuracy} |
| 5-fold CV f1_macro (best C) | {cv_f1} |
| Train rows | {n_train:,} |
| Validation rows | {n_val:,} |

Classes: `{classes}`.

## Honest framing — weak supervision

Training labels come from the regex extractors in
[`ingestion/normalize.py`](https://github.com/Arjun10g/na-tech-jobs/blob/main/ingestion/normalize.py).
Specifically:

- Rows where the regex matched a specific keyword get the explicit label.
- Rows where the regex *defaulted* (e.g. `"mid"` for unmatched seniority
  titles, `"Other"` for unmatched role-family titles) are **dropped from
  training** — that fallback signal is too noisy to teach from.

The model's job is to **generalize** the regex via the encoder's semantic
embedding space — it should classify titles like "ML Researcher" correctly
even though the regex didn't match them. This means:

1. Eval metrics here measure **agreement with the regex** on a held-out
   slice — they don't measure agreement with a hand-labeled gold standard.
2. CLAUDE.md §7 calls for a hand-labeled clean test set of 500 examples
   for proper evaluation. v2 will land this when capacity allows.
3. Confidence scores (`predict_proba`) are useful for filtering high-
   confidence predictions in downstream consumers.

## Class balance during training

```
{json.dumps(label_counts, indent=2)}
```

## Inputs

- `title` and `description_md` (truncated to first 1,000 chars) joined
  with `" — "` and embedded by the frozen encoder.

## Inference

Direct (sklearn + sentence-transformers):

```python
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import joblib

local_dir = snapshot_download("arjun10g/na-tech-jobs-{name}-v1")
artifact = joblib.load(f"{{local_dir}}/classifier.joblib")
encoder = SentenceTransformer(artifact["encoder_id"])
clf = artifact["classifier"]
id2label = artifact["id2label"]

text = "Senior Machine Learning Engineer — We're hiring an MLE…"
emb = encoder.encode([text], normalize_embeddings=True)
pred_id = int(clf.predict(emb)[0])
print(id2label[pred_id])
```

Or via the project's wrapper class:

```python
from models.{name}.predict import {("Seniority" if name == "seniority" else "RoleFamily")}Classifier
clf = {("Seniority" if name == "seniority" else "RoleFamily")}Classifier.load_from_hub()
clf.predict(["Senior ML Engineer at Stripe"])
```

{prelim_block}

## Citation

> Ghumman, A. (2026). _na-tech-jobs {pretty_name} classifier v1._
> https://huggingface.co/arjun10g/na-tech-jobs-{name}-v1
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("classifier", choices=("seniority", "role_family"))
    p.add_argument("--artifacts-dir", default=None, help="Defaults to data/models/<classifier>")
    p.add_argument(
        "--create", action="store_true", help="Create the HF Model repo if it doesn't exist"
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    from huggingface_hub import create_repo, upload_folder

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    artifacts_dir = Path(args.artifacts_dir or f"data/models/{args.classifier}")
    final_dir = artifacts_dir / "final"
    summary_path = artifacts_dir / "training_summary.json"
    if not final_dir.exists():
        raise FileNotFoundError(f"missing trained model at {final_dir}")

    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    repo_id = f"arjun10g/na-tech-jobs-{args.classifier}-v1"
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    if args.create:
        try:
            create_repo(repo_id, repo_type="model", token=token, exist_ok=True)
            logger.info("ensured model repo %s", repo_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("create_repo failed: %s", exc)

    readme_path = final_dir / "README.md"
    readme_path.write_text(_model_card(args.classifier, summary))

    if summary_path.exists():
        artifact_summary = final_dir / "training_summary.json"
        artifact_summary.write_text(summary_path.read_text())

    upload_folder(
        repo_id=repo_id,
        folder_path=str(final_dir),
        repo_type="model",
        token=token,
        commit_message=f"v1 :: {args.classifier} classifier (eval={summary.get('eval', {})})",
    )
    logger.info("pushed %s to %s", final_dir, repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
