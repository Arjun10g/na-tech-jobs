"""Generate the HF Hub model card for the winning tier.

The card follows [Mitchell et al 2019, _Model Cards for Model Reporting_,
FAccT] and CLAUDE.md §7's "honest framing" requirement: the model predicts
salary as priced by **disclosing employers in our corpus**, NOT ground
truth. Selection-bias / MNAR caveats land in §3 of the card.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CARD_TEMPLATE = """\
---
license: mit
library_name: scikit-learn
tags:
- regression
- salary-prediction
- north-america
- tabular
metrics:
- mae
- mape
- r2
model-index:
- name: na-tech-jobs-salary-v1
  results:
  - task:
      type: regression
      name: Salary Regression
    dataset:
      name: arjun10g/na-tech-jobs (curated/jobs.parquet)
      type: arjun10g/na-tech-jobs
    metrics:
    - type: mae
      name: MAE (USD/year)
      value: {mae:.0f}
    - type: mape
      name: MAPE (%)
      value: {mape_pct:.2f}
    - type: r2
      name: R² (log scale)
      value: {r2_log:.4f}
---

# na-tech-jobs salary regressor — v1

Predicts the **maximum disclosed salary** of a North American senior tech
job posting, in USD per year, given tabular features from the
[`arjun10g/na-tech-jobs`](https://huggingface.co/datasets/arjun10g/na-tech-jobs)
weekly snapshot.

## 1. Headline metrics

| Metric | Test-set value | 95% bootstrap CI | 5-fold CV-OOF on train | CV 95% bootstrap CI |
|---|---|---|---|---|
| MAE (USD / year) | **${mae:,.0f}** | ${mae_ci_low:,.0f} – ${mae_ci_high:,.0f} | {cv_mae_str} | {cv_mae_ci_str} |
| MAPE | {mape_pct:.2f}% | — | {cv_mape_str} | — |
| R² (on log10 target) | {r2_log:.4f} | — | {cv_r2_str} | — |
| n | {n_test:,} | (frozen test, stratified by `country × source`) | {cv_n_str} | (5-fold OOF) |

The CLAUDE.md §7 target was **MAE < $25k USD/year**.
{target_status}

The two columns answer different questions: **test-MAE** is generalization to
the frozen 20% holdout (a single draw); **CV-MAE** is the average of 5
out-of-fold MAEs on the training set, capturing **split variance**. When the
two agree we have evidence the test draw was representative; if test-MAE
materially differs from CV-MAE we'd suspect either a lucky test draw or
overfitting.

## 2. The ladder

We trained six tiers from a constant baseline up to XGBoost, all on the
**same frozen test set**. The selected model is `{winning_tier}`.

{leaderboard_md}

The leaderboard answers the question "is the gain from XGBoost worth its
complexity?" — read down the bootstrap CIs to see where they overlap.

## 3. Honest framing — selection bias & MNAR

The model is trained on the **disclosed-salary subset only** (~50% of NA
tech postings). Disclosure is **not random**:

- **Mandated jurisdictions** (CA / NY / WA / CO / CT / MD / IL / HI;
  ON / BC / PEI) have higher disclosure rates by law.
- **Voluntary disclosure** is concentrated at transparency-leaning employers
  (Stripe, Anthropic, Anduril, Cohere over-represent the disclosed sample).
- **Strategic non-disclosure** (Cullen & Pakzad-Hurson 2023) means the
  unobserved component depends on the latent salary itself — i.e. the
  process is MNAR.

**Therefore the model predicts "salary as priced by disclosing employers
in our corpus", not ground truth.** A 2-stage Heckman correction is
flagged in `LITERATURE_REVIEW.md` §1.2 as a v2 deliverable.

## 4. Inputs

Tabular features (no text yet — `description_md` enters via bge-m3
embeddings in Phase 5). See
[`DATA_DICTIONARY.md`](https://github.com/Arjun10g/na-tech-jobs/blob/main/DATA_DICTIONARY.md)
for full definitions and
[`LITERATURE_REVIEW.md` §14](https://github.com/Arjun10g/na-tech-jobs/blob/main/LITERATURE_REVIEW.md)
for encoding choices.

## 5. Stratified evaluation

Per-stratum MAE on the test set ({n_test:,} rows total):

{stratified_md}

CA strata are small (~few hundred rows); narrow CI claims on CA
performance should be read with that caveat. The choice to **emphasise
uncertainty directly** via bootstrap CIs (rather than a Cohen-style power
analysis) is documented in `LITERATURE_REVIEW.md` §15.3 #15.

## 6. Intended use

- **Recruiters and candidates**: rough salary anchoring for a given role
  + location + seniority bucket.
- **The builder**: their own NA senior-DS job search.
- **Researchers**: a transparent baseline for compensation prediction
  using only public ATS data.

## 7. Out-of-scope

- Non-NA job markets (the dataset is US/CA only).
- Non-tech sectors (banks, healthcare, retail are largely on Workday
  and not yet in the dataset; Phase 4 of the project plan).
- Total compensation (the target is base salary max; equity / bonus
  are mentioned only as boolean features).
- Individual-offer negotiation (the model predicts a posting, not an
  offer).

## 8. Training data

- Source: weekly ingest from Greenhouse, Lever, Ashby ATS APIs.
- Snapshot: {snapshot_date}.
- Total active rows: 12,334. Disclosed-salary subset: ~6,143.
- Train / test split: deterministic, hash-keyed by `id` and seed=42,
  stratified by `(country, source)`. 80/20 split. Test row IDs frozen
  at `data/eda/test_split_ids.json`.

## 9. Limitations

- **Workday gap** — major employers (Snowflake, Coinbase, Shopify,
  Etsy, Wayfair, DoorDash) use Workday, which our extractor doesn't
  yet support. Their salaries are missing from training data.
- **No total-comp signal** — the regressor sees `offers_equity`
  (boolean) and `bonus_mentioned` (boolean) but cannot distinguish a
  $200k base + $100k equity package from $300k cash.
- **Title-derived seniority/role family is regex-noisy** — ~70% of
  rows label as `Other` for role family. Phase 4 will replace these
  with DeBERTa classifiers.
- **Description text not yet used** — bge-m3 dense embedding lands in
  Phase 5. The current model is purely tabular.

## 10. Reproducibility

```bash
git clone https://github.com/Arjun10g/na-tech-jobs
cd na-tech-jobs
uv sync --extra ml --extra eda --group dev
uv run python -m models.salary.train
```

This will rebuild the dataset, re-run the ladder, refit the winning
tier on the same frozen split, and reproduce the metrics above. Random
seeds are fixed throughout (`42` for split, Optuna sampler, RF, XGB).

## Citation

> Ghumman, A. (2026). _na-tech-jobs salary regressor v1._
> https://huggingface.co/arjun10g/na-tech-jobs-salary-v1
"""


def _stratified_md(stratified: pd.DataFrame) -> str:
    if stratified.empty:
        return "_No per-stratum breakdown (all strata had < 5 rows)._"
    df = stratified.copy()
    df["mae"] = df["mae"].round(0).astype(int).map(lambda x: f"${x:,}")
    df["mae_ci_low"] = df["mae_ci_low"].round(0).astype(int).map(lambda x: f"${x:,}")
    df["mae_ci_high"] = df["mae_ci_high"].round(0).astype(int).map(lambda x: f"${x:,}")
    df["mape_pct"] = df["mape_pct"].round(2)
    df["r2_log"] = df["r2_log"].round(4)
    return df[
        ["stratum", "n", "mae", "mae_ci_low", "mae_ci_high", "mape_pct", "r2_log"]
    ].to_markdown(index=False)


def _leaderboard_md(report: dict[str, Any]) -> str:
    rows = report["tiers"]
    df = pd.DataFrame(rows)
    df["mae"] = df["mae"].astype(int).map(lambda x: f"${x:,}")
    df["mae_ci"] = df.apply(
        lambda r: f"${int(r['mae_ci_low']):,}-${int(r['mae_ci_high']):,}", axis=1
    )
    if "cv_mae" in df.columns:
        df["cv_mae"] = (
            df["cv_mae"].fillna(-1).astype(int).map(lambda x: f"${x:,}" if x >= 0 else "—")
        )
        df["cv_mae_ci"] = df.apply(
            lambda r: (
                f"${int(r['cv_mae_ci_low']):,}-${int(r['cv_mae_ci_high']):,}"
                if pd.notna(r.get("cv_mae_ci_low"))
                else "—"
            ),
            axis=1,
        )
    df["mape_pct"] = df["mape_pct"].round(2)
    df["r2_log"] = df["r2_log"].round(4)
    cols = ["tier", "mae", "mae_ci"]
    if "cv_mae" in df.columns:
        cols += ["cv_mae", "cv_mae_ci"]
    cols += ["mape_pct", "r2_log", "n_test"]
    return df[cols].to_markdown(index=False)


def render_model_card(
    report: dict[str, Any],
    winning_result: dict[str, Any],
    stratified: pd.DataFrame,
    snapshot_date: str,
) -> str:
    target_status = (
        f"Achieved: ${winning_result['mae']:,.0f} ✓"
        if winning_result["mae"] < 25_000
        else f"Not yet hit (current: ${winning_result['mae']:,.0f})."
    )
    cv_mae = winning_result.get("cv_mae")
    cv_mae_lo = winning_result.get("cv_mae_ci_low")
    cv_mae_hi = winning_result.get("cv_mae_ci_high")
    cv_mape = winning_result.get("cv_mape_pct")
    cv_r2 = winning_result.get("cv_r2_log")
    cv_n = report.get("n_train")
    return CARD_TEMPLATE.format(
        mae=winning_result["mae"],
        mae_ci_low=winning_result["mae_ci_low"],
        mae_ci_high=winning_result["mae_ci_high"],
        mape_pct=winning_result["mape_pct"],
        r2_log=winning_result["r2_log"],
        n_test=winning_result["n_test"],
        winning_tier=winning_result["tier"],
        target_status=target_status,
        leaderboard_md=_leaderboard_md(report),
        stratified_md=_stratified_md(stratified),
        snapshot_date=snapshot_date,
        cv_mae_str=f"**${cv_mae:,.0f}**" if cv_mae is not None else "_n/a_",
        cv_mae_ci_str=(
            f"${cv_mae_lo:,.0f} – ${cv_mae_hi:,.0f}"
            if cv_mae_lo is not None and cv_mae_hi is not None
            else "_n/a_"
        ),
        cv_mape_str=f"{cv_mape:.2f}%" if cv_mape is not None else "_n/a_",
        cv_r2_str=f"{cv_r2:.4f}" if cv_r2 is not None else "_n/a_",
        cv_n_str=f"{cv_n:,}" if cv_n is not None else "_n/a_",
    )


def write_model_card(
    report: dict[str, Any],
    stratified: pd.DataFrame,
    out_path: Path,
    snapshot_date: str | None = None,
) -> Path:
    if snapshot_date is None:
        snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    winning_name = report["winning_tier"]
    winning_result = next(t for t in report["tiers"] if t["tier"] == winning_name)
    body = render_model_card(report, winning_result, stratified, snapshot_date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    return out_path
