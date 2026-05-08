"""Shared training + inference plumbing for the title-text classifiers
(seniority, role-family).

v1 architecture (per LITERATURE_REVIEW.md §17): **frozen sentence-transformer
embeddings + multinomial logistic regression head**. We don't fine-tune.

- Why: short-text small-vocabulary classification with weakly supervised
  labels does not benefit from full-encoder fine-tuning (Peters et al 2019,
  "To Tune or Not to Tune?"; Tunstall et al 2022, SetFit). A linear probe
  on a strong general-purpose embedder reaches the same operating point at
  ~100x less compute.
- Encoder: ``sentence-transformers/all-MiniLM-L6-v2`` (22 M params, 384-dim).
  Picked over bge-m3 (568 M, 1024-dim) for v1 because the marginal F1 lift
  from a heavier encoder is small relative to the wall-clock cost on CPU /
  MPS, and we want training to be reproducible in <1 minute.
- Classifier: sklearn ``LogisticRegression`` with multinomial loss, lbfgs
  solver, L2 penalty, class-weight balanced. C is selected by 5-fold CV
  over ``{0.1, 1, 10}``.
- Labels: regex-confident only — rows where the regex defaulted (``"mid"``
  for seniority, ``"Other"`` / ``"Manager"`` for role family) are dropped.
- Eval: stratified 10% holdout for headline metrics, plus 5-fold CV for the
  C-search and bootstrap 95% CI on F1-macro.
- Artifact: a single ``classifier.joblib`` containing the fitted LR plus
  ``encoder_id`` / ``label2id`` / ``id2label`` / metadata; the encoder is
  re-instantiated by id at load time (we don't ship the 22 M-param weights
  ourselves — sentence-transformers caches them locally on first use).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("models.classifier_base")

DEFAULT_ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_DIR = Path("data/models")
DEFAULT_C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
DEFAULT_TEXT_TRUNCATE = 1000  # chars of description_md to keep
ARTIFACT_FILENAME = "classifier.joblib"


@dataclass
class ClassifierSpec:
    """Per-classifier configuration."""

    name: str  # e.g. "seniority", "role_family"
    label_column: str  # the column on curated/jobs.parquet with regex labels
    drop_labels: tuple[str, ...]  # regex-default labels to drop from train (e.g. "mid", "Other")
    text_columns: tuple[str, ...] = ("title", "description_md")
    text_separator: str = " — "

    @property
    def hf_repo_id(self) -> str:
        return f"arjun10g/na-tech-jobs-{self.name}-v1"

    @property
    def output_dir(self) -> Path:
        return DEFAULT_OUTPUT_DIR / self.name


# ── Dataset prep ──────────────────────────────────────────────────────────


def build_classifier_dataset(
    spec: ClassifierSpec,
    curated_path: Path = Path("data/curated/jobs.parquet"),
    *,
    test_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    """Materialize train / test text+label arrays with stratified sampling
    on the kept (non-default) labels."""
    df = pd.read_parquet(curated_path)
    logger.info("loaded %d rows from %s", len(df), curated_path)

    parts: list[pd.Series] = []
    for col in spec.text_columns:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
    if not parts:
        raise ValueError(f"no text columns from {spec.text_columns} found")
    title = parts[0]
    description = parts[1] if len(parts) > 1 else pd.Series([""] * len(df))
    text = title + spec.text_separator + description.str[:DEFAULT_TEXT_TRUNCATE]
    df = df.assign(_text=text)

    confident_mask = df[spec.label_column].notna() & ~df[spec.label_column].isin(spec.drop_labels)
    df_train_pool = df.loc[confident_mask].copy()
    logger.info(
        "%s :: %d/%d rows have confident regex labels (kept) — dropped %d defaults",
        spec.name,
        len(df_train_pool),
        len(df),
        len(df) - len(df_train_pool),
    )

    labels = sorted(df_train_pool[spec.label_column].unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    df_train_pool["_label_id"] = df_train_pool[spec.label_column].map(label2id)

    rng = np.random.default_rng(seed)
    val_idx_list: list[int] = []
    for _, group in df_train_pool.groupby(spec.label_column):
        n_val = max(1, int(round(len(group) * test_frac)))
        chosen = rng.choice(group.index.values, size=n_val, replace=False)
        val_idx_list.extend(chosen)
    val_set = set(val_idx_list)
    train_df = df_train_pool[~df_train_pool.index.isin(val_set)]
    val_df = df_train_pool[df_train_pool.index.isin(val_set)]

    label_counts = df_train_pool[spec.label_column].value_counts().to_dict()
    logger.info("%s class balance :: %s", spec.name, label_counts)

    return {
        "train_texts": train_df["_text"].tolist(),
        "train_labels": train_df["_label_id"].tolist(),
        "val_texts": val_df["_text"].tolist(),
        "val_labels": val_df["_label_id"].tolist(),
        "label2id": label2id,
        "id2label": id2label,
        "labels_sorted": labels,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_total_curated": len(df),
        "n_dropped_defaults": int((~confident_mask).sum()),
        "label_counts": label_counts,
    }


# ── Encoder cache ─────────────────────────────────────────────────────────


_ENCODER_CACHE: dict[str, Any] = {}


def _load_encoder(encoder_id: str):
    if encoder_id in _ENCODER_CACHE:
        return _ENCODER_CACHE[encoder_id]
    import torch
    from sentence_transformers import SentenceTransformer

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info("loading encoder %s on %s", encoder_id, device)
    model = SentenceTransformer(encoder_id, device=device)
    _ENCODER_CACHE[encoder_id] = model
    return model


def encode_texts(texts: list[str], encoder_id: str, *, batch_size: int = 64) -> np.ndarray:
    """Mean-pooled, L2-normalized embeddings."""
    model = _load_encoder(encoder_id)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 500,
        convert_to_numpy=True,
    )
    return np.asarray(emb, dtype=np.float32)


# ── Training ──────────────────────────────────────────────────────────────


def _bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI on macro-F1."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return float(lo), float(hi)


def train_classifier(
    spec: ClassifierSpec,
    *,
    encoder_id: str = DEFAULT_ENCODER_ID,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    cv_folds: int = 5,
    seed: int = 42,
    output_dir: Path | None = None,
    # Legacy kwargs accepted for CLI compatibility — silently ignored.
    epochs: int | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Train a frozen-encoder + LR classifier. Returns training summary."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if model_name and model_name != encoder_id:
        # The CLI flag is named --model-name for backward-compat with the
        # earlier DeBERTa-LoRA script. Honor it as the encoder id.
        logger.info("using --model-name override as encoder id: %s", model_name)
        encoder_id = model_name

    output_dir = output_dir or spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = build_classifier_dataset(spec, seed=seed)
    n_classes = len(data["label2id"])

    logger.info(
        "%s :: encoding %d train + %d val texts with %s",
        spec.name,
        data["n_train"],
        data["n_val"],
        encoder_id,
    )
    X_train = encode_texts(data["train_texts"], encoder_id)
    X_val = encode_texts(data["val_texts"], encoder_id)
    y_train = np.asarray(data["train_labels"], dtype=np.int64)
    y_val = np.asarray(data["val_labels"], dtype=np.int64)
    logger.info("embedding shape :: train=%s val=%s", X_train.shape, X_val.shape)

    # 5-fold CV on the train pool to pick C.
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_results: dict[float, dict[str, float]] = {}
    best_c = c_grid[0]
    best_score = -np.inf
    for c in c_grid:
        clf = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )
        scores = cross_val_score(
            clf,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        mean = float(scores.mean())
        std = float(scores.std())
        cv_results[c] = {"f1_macro_mean": mean, "f1_macro_std": std}
        logger.info("CV C=%.3g :: f1_macro = %.4f ± %.4f", c, mean, std)
        if mean > best_score:
            best_score = mean
            best_c = c
    logger.info("selected C=%.3g (CV f1_macro=%.4f)", best_c, best_score)

    # Final fit on full train pool with the chosen C.
    final_clf = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    final_clf.fit(X_train, y_train)

    # Headline metrics on the held-out val split.
    val_preds = final_clf.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1_macro = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_f1_weighted = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))
    f1_lo, f1_hi = _bootstrap_f1_ci(y_val, val_preds)

    report = classification_report(
        y_val,
        val_preds,
        target_names=data["labels_sorted"],
        digits=3,
        zero_division=0,
        output_dict=True,
    )

    # Save artifact.
    save_dir = output_dir / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "encoder_id": encoder_id,
        "classifier": final_clf,
        "label2id": data["label2id"],
        "id2label": data["id2label"],
        "labels_sorted": data["labels_sorted"],
        "n_features": int(X_train.shape[1]),
        "best_C": float(best_c),
        "spec_name": spec.name,
        "version": "v1",
    }
    artifact_path = save_dir / ARTIFACT_FILENAME
    joblib.dump(artifact, artifact_path)
    logger.info(
        "saved artifact :: %s (%.1f KB)", artifact_path, artifact_path.stat().st_size / 1024
    )

    summary = {
        "classifier": spec.name,
        "architecture": "frozen-sentence-transformer + logistic-regression",
        "encoder_id": encoder_id,
        "n_classes": n_classes,
        "n_features": int(X_train.shape[1]),
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "n_dropped_defaults": data["n_dropped_defaults"],
        "label_counts": data["label_counts"],
        "id2label": data["id2label"],
        "label2id": data["label2id"],
        "cv": {
            "folds": cv_folds,
            "c_grid": list(c_grid),
            "results": {str(c): v for c, v in cv_results.items()},
            "best_C": float(best_c),
            "best_f1_macro": float(best_score),
        },
        "eval": {
            "eval_accuracy": round(val_acc, 4),
            "eval_f1_macro": round(val_f1_macro, 4),
            "eval_f1_weighted": round(val_f1_weighted, 4),
            "eval_f1_macro_ci95": [round(f1_lo, 4), round(f1_hi, 4)],
        },
        "classification_report": report,
        "save_dir": str(save_dir),
        "artifact_path": str(artifact_path),
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "%s :: training done. val acc=%.3f f1_macro=%.3f (95%% CI [%.3f, %.3f])",
        spec.name,
        val_acc,
        val_f1_macro,
        f1_lo,
        f1_hi,
    )
    return summary
