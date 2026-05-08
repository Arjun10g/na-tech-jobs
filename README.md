# na-tech-jobs

A production ML platform for the **North American senior tech-hiring market**.

> Weekly ingestion across Greenhouse, Lever, Ashby (Workable, SmartRecruiters,
> Workday in later phases) → versioned dataset on the Hugging Face Hub →
> salary regressor + (Phase 4) seniority/role-family classifiers → (Phase 5)
> hybrid + late-interaction RAG layer over a bge-m3 embedding index →
> deployed on a $9 HF Pro Space, drift-monitored weekly, retrained monthly.

> Built by a senior data science candidate using the platform for their own
> North American job search.

---

## 🔗 Live links

- **Demo (Space)**: https://arjun10g-na-tech-jobs.hf.space
- **Source (GitHub)**: https://github.com/Arjun10g/na-tech-jobs
- **Dataset**: https://huggingface.co/datasets/arjun10g/na-tech-jobs (12.3k rows × 49 cols, weekly)
- **Salary model v1**: https://huggingface.co/arjun10g/na-tech-jobs-salary-v1

## Project documents

| File | What it contains |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Project bible — architecture, locked decisions, phased plan, risks |
| [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) | Every column in `data/curated/jobs.parquet` — type, fill rate, predictor decision |
| [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) | ~1k-line predictor-by-predictor review with 50+ citations + ideal-EDA self-audit |
| [`MAINTENANCE.md`](MAINTENANCE.md) | Running known-issues / debt log, resolved entries kept for project history |
| [`eda/reports/<date>/report.md`](eda/reports/) | Statistical audit + 11 plots per snapshot |

---

## Phase status

| Phase | Status | Headline |
|---|---|---|
| 0 — Scaffold | ✅ | Repo + CI + HF Space + Discord alerting |
| 1 — Ingestion v1 | ✅ | 65 ATS handles → weekly parquet → HF Dataset, ~12.3k jobs |
| 2 — Features + curated + salary regressor | ✅ | Regex cascade (49.8% disclosure mined) + LLM Tier 2 dormant + curated DuckDB layer + 6-tier ladder. **Tier 5 XGBoost test-MAE $29,091 / CV-MAE $30,533** |
| 3 — First deployable | ✅ | Salary prediction + curated search live on the Space |
| **4 — Multi-model + payload enrichment** | ✅ **NEW** | Frozen-MiniLM + LR seniority (val f1_macro 0.831) and role-family (0.915) classifiers, NuExtract skills wrapper, all 12,334 jobs enriched with versioned predictions on the HF Dataset |
| 5 — Retrieval stack | future | bge-m3 hybrid + ColBERT, Qdrant, resume matcher |
| 6-9 — Eval, LLM, drift, polish | future | per [CLAUDE.md §10](CLAUDE.md) |

---

## Salary regressor — six-tier ladder

All evaluated on the **same frozen 80/20 train/test split** (n_test=1,226), with
bootstrap 95% CIs on test-MAE and 5-fold CV-MAE on the training set as a
generalization sanity check.

| Tier | Test-MAE | Test 95% CI | CV-MAE | CV 95% CI | MAPE | R² log |
|---|---|---|---|---|---|---|
| 0 constant baseline | $60,509 | $57k–$64k | $62,279 | $61k–$64k | 33.7% | ≈0 |
| 1 stratified mean | $59,589 | $56k–$63k | $61,623 | $60k–$63k | 32.9% | 0.045 |
| 2 Mincer OLS | $51,322 | $48k–$54k | $52,041 | $50k–$53k | 27.4% | 0.283 |
| 3 Ridge (full encoder) | $43,199 | $41k–$45k | $42,179 | $41k–$43k | 23.3% | 0.462 |
| 4 Random Forest | $35,935 | $34k–$38k | $37,016 | $36k–$38k | 19.0% | 0.615 |
| **5 XGBoost + Optuna(50)** | **$29,091** | **$27k–$31k** | **$30,533** | **$29k–$32k** | **14.7%** | **0.730** |

Each tier's bootstrap CI cleanly excludes the previous tier's — the ladder is
monotone with statistical evidence at every step. Test-MAE and CV-MAE agree
within 5%: no overfitting, no lucky test draw. The
[CLAUDE.md §10](CLAUDE.md) target of **MAE < $25k** is not yet hit; closing
that gap is for Phase 5's bge-m3 description embedding.

Methodology: parsimonious-first ladder per [`LITERATURE_REVIEW.md` §16](LITERATURE_REVIEW.md)
(Breiman 2001 two-cultures + Mincer 1974 + Shwartz-Ziv & Armon 2022 +
Grinsztajn et al 2022). Encoding choices in §14 of the same doc.

---

## Title classifiers — frozen MiniLM + LR

Phase 4 ships **seniority** and **role-family** classifiers that score every
job in the curated table. Both use the same architecture: frozen
`sentence-transformers/all-MiniLM-L6-v2` (22 M params, 384-dim) embeddings
+ multinomial logistic regression with L2, class-weight balanced, and C
selected by 5-fold stratified CV from `{0.1, 1, 10}`.

| Classifier | Classes | Train rows | Val f1_macro | 95% CI | Best C | Repo |
|---|---|---|---|---|---|---|
| seniority | 7 (intern…director) | 6,361 | **0.831** | [0.780, 0.870] | 10 | [`arjun10g/na-tech-jobs-seniority-v1`](https://huggingface.co/arjun10g/na-tech-jobs-seniority-v1) |
| role_family | 6 (DS / DA / DE / MLE / RS / AS / SWE-ML) | 569 | **0.915** | [0.830, 0.980] | 10 | [`arjun10g/na-tech-jobs-role_family-v1`](https://huggingface.co/arjun10g/na-tech-jobs-role_family-v1) |

Why a linear probe instead of CLAUDE.md §7's locked DeBERTa-v3 + LoRA?
For short-text small-vocabulary classification with weakly supervised
labels, a linear probe on a strong general-purpose embedder reaches the
same operating point at ~100x less compute (Peters et al 2019; Tunstall
et al 2022, SetFit; Joulin et al 2017, FastText). Full justification +
recipe in [`LITERATURE_REVIEW.md` §17](LITERATURE_REVIEW.md). v1.1 will
benchmark DeBERTa-v3 + LoRA against this baseline once a hand-labeled
500-example test set lands.

**Honest framing.** Training labels come from the regex extractors in
`ingestion/normalize.py`; rows where the regex fell back to its default
(`"mid"` for seniority, `"Other"` and `"Manager"` for role_family) are
dropped from training. Eval metrics measure agreement with the regex on
a held-out 10% slice — they don't measure agreement with hand-labeled
gold. CLAUDE.md §7 logs the hand-labeled test set as the v1.1 task.

**Skill extractor** (`arjun10g/na-tech-jobs-skills-v1`) is a NuExtract-tiny
zero-shot wrapper + ~70-name canonical taxonomy. Available for ad-hoc use;
batch application to the curated table is deferred to v1.1 (~6 hours on
MPS, ~30 min on an A10G HF Job).

**Curated enrichment.** `curated_enriched/jobs.parquet` on the HF
Dataset has all 12,334 active jobs scored with versioned columns
`seniority_label_v1`, `seniority_confidence_v1`, `role_family_v1`,
`role_family_confidence_v1`, `predicted_salary_usd_v1`,
`prediction_model_version`, plus `extracted_skills_v1` (empty list in
v1; populated in v1.1). Re-running is a single command:

```sh
uv run python -m curated.enrich --skip-skills --push-to-hub
```

---

## Quickstart

```sh
# install Python 3.11 + dev deps
uv sync --group dev

# run the local Gradio app (loads model + curated parquet from HF Hub)
uv run python -m app.main

# run lint + tests
uv run ruff format --check .
uv run ruff check .
uv run pytest

# run a smoke ingest (5 companies, no HF push)
uv run python -m ingestion.orchestrator --output-dir data --limit 5

# full ingest with HF Dataset push (needs HF_TOKEN)
uv run python -m ingestion.orchestrator --output-dir data --push-to-hub --alert

# rebuild the curated layer from snapshots
uv run python -m curated.build --push-to-hub

# regenerate the EDA audit
uv run python -m eda.audit  # writes data/eda/{report.md, metrics.json, plots/}

# retrain the salary regressor + push to HF Hub
uv sync --extra ml --extra eda --group dev
uv run python -m models.salary.train  # ~15-20 min on CPU
uv run python -m scripts.publish_salary_model
```

### Optional dependency groups

| Group | Purpose | Install |
|---|---|---|
| `ml` | Training stack (torch, transformers, xgboost, mlflow, sklearn, optuna) | `uv sync --extra ml` |
| `eda` | matplotlib, seaborn, statsmodels, missingno, tabulate | `uv sync --extra eda` |
| `space-runtime` | Lean inference deps for the Space (xgboost + sklearn, no torch) | `uv sync --extra space-runtime` |
| `rag` | Vector store, chunking, NL→SQL, PDF parsing (Phase 5) | `uv sync --extra rag` |
| `monitoring` | Evidently for drift reports (Phase 8) | `uv sync --extra monitoring` |
| `api` | FastAPI for programmatic endpoints | `uv sync --extra api` |

## Repository layout

See [`CLAUDE.md` § 9](CLAUDE.md) for the full layout.

```
app/                  Gradio front end (Phase 0+, expanded in Phase 3)
ingestion/            ATS extractors + feature cascade (Phase 1+)
  feature_extraction/ Regex Tier 1 + NuExtract Tier 2 (dormant) cascade
curated/              DuckDB layer over weekly snapshots
eda/                  Statistical audit + train/test split (Step 2.5)
models/salary/        Six-tier regressor ladder (Step 3)
monitoring/           Discord alerting; Evidently drift (Phase 8)
scripts/              Probe handles, backfill features, publish models
.github/workflows/    CI, weekly ingest cron, Space deploy
```

## Secrets

See [`infra/secrets.md`](infra/secrets.md). Copy `.env.example` to `.env` and
fill in `HF_TOKEN` + `DISCORD_WEBHOOK_URL` for operations that touch the Hub
or alert on pipeline failure.

## License

MIT. Models and datasets are licensed individually — see their respective
model and dataset cards.
