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
| 4 — Multi-model + payload enrichment | ✅ | Frozen-MiniLM + LR seniority (val f1_macro 0.812 reviewed-gold) and role-family (0.934) classifiers, regex skills layer, all 12,334 jobs enriched with versioned predictions on the HF Dataset |
| 5 — Retrieval stack | ✅ | Parent-child chunking (29k parents, 120k children) + Qdrant local-mode + dense (MiniLM 384-dim) hybrid pipeline + cross-encoder rerank (optional) + Matcher tab live. bge-m3 reindex queued as v1.1 |
| 6a — Retrieval eval harness | ✅ | 48 labeled retrieval queries + recall@k / MRR / nDCG@10 metrics. `hybrid+rerank` recall@10 = **0.486** (vs `dense` 0.363). HyDE + ColBERT toggles land after the bge-m3 reindex |
| 7 — NL→SQL analytics | ✅ | Natural-language → DuckDB SQL with mandatory sqlglot safety layer (CLAUDE.md §11): allowlisted tables/columns, DDL/multi-statement reject, 1000-row + 5-s caps. Anthropic / HF Inference / mock LLM backends. Analytics tab live, executed SQL always shown. **61 dedicated safety tests.** |
| **8a — Operational dashboard** | ✅ **NEW** | Evidently drift detection between two snapshots (PSI ≥ 0.20 → priority retrain), pipeline-health rollup (per-extractor success/fail), market-trend tabs (salary distribution by role × seniority, top companies, role-family share by country, top skills) all live in the new Dashboard tab. Live numbers: 12,334 active jobs, 49.8% disclosure, top company **Anduril (1,888 postings)**, Canada SWE-ML share 29.6% vs US 21.2%. |
| **8b — CI workflows** | ✅ **NEW** | `.github/workflows/drift.yml` (Mondays 03:00 UTC, pulls 4-week-old reference + latest snapshot, runs Evidently, pushes report to dataset repo, alerts Discord on PSI breach) + `.github/workflows/retrain.yml` (monthly 1st @ 04:00 UTC, matrix over classifiers, champion/challenger gate via `monitoring.champion_challenger`, conditional publish if challenger passes) |
| 9 — polish | future | per [CLAUDE.md §10](CLAUDE.md) |

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
dropped from training. The held-out F1 above measures agreement with the
regex on a held-out 10% slice. To check the classifier didn't just
memorize the regex, we also ran a **two-pass Claude-reviewed eval set**
on a 230-row stratified sample per classifier (`eval/<classifier>_test.jsonl`):
first-pass labelers produced LLM proposals; second-pass reviewers
were shown those proposals plus the classifier's prediction and either
accepted (1.7% override rate) or corrected on title-vs-label
contradictions:

| Classifier | vs regex (held-out) | vs reviewed gold (in-vocab) | 95% CI |
|---|---|---|---|
| seniority | 0.831 | **0.812** | [0.7347, 0.8729] |
| role_family | 0.915 | **0.934** | [0.8761, 0.9765] |

The "in-vocab" filter excludes the ~49% of sampled rows where the gold
label is `"mid"` / `"Other"` / `"Manager"` — labels we drop from training.
The classifier is a *specialist over the explicit labels*, not a
general-purpose 9-way classifier; production callers should use
confidence thresholds + the regex's default-label flag to decide when to
trust the prediction. Two-pass Claude review is still a higher-quality
proxy than full *human* review — that v1.2 task uses the same flow:
`scripts.label_classifier --review` shows each row with the LLM proposal
and lets the user accept / override.

**Skill extractor** is **regex-first** by default — `extracted_skills_v1`
is populated from the existing `ingestion/feature_extraction/regex/tech_stack.py`
column on every weekly ingest (free, deterministic, ~ms, 64.7% coverage
across the 12,334 active jobs). The NuExtract LLM tier
(`arjun10g/na-tech-jobs-skills-v1`) stays opt-in via
`--skills-mode=nuextract` and runs during monthly retrains on an A10G
(CLAUDE.md §10's cadence; logged in MAINTENANCE.md).

**Curated enrichment.** `curated_enriched/jobs.parquet` on the HF
Dataset has all 12,334 active jobs scored with versioned columns
`seniority_label_v1`, `seniority_confidence_v1`, `role_family_v1`,
`role_family_confidence_v1`, `predicted_salary_usd_v1`,
`prediction_model_version`, plus `extracted_skills_v1` (regex-populated,
64.7% coverage). Re-running is a single command:

```sh
uv run python -m curated.enrich --push-to-hub
```

---

## Hybrid retrieval — Phase 5 matcher

Phase 5 ships a parent-child chunked Qdrant index over the curated job
corpus and a Matcher tab that does natural-language → ranked-jobs
retrieval.

| | Detail |
|---|---|
| Chunking | parent-child `RecursiveCharacterTextSplitter` (~1024-token parents, ~256-token / 32-overlap children, hierarchical markdown separators) |
| Volume indexed | **12,334 jobs → 29,311 parents → 120,004 children** |
| Embedder (v1) | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, dense-only). bge-m3 (1024-dim dense + sparse + ColBERT multivec) wired but reindex deferred to v1.1 — ~8 hr on Apple MPS or ~1 hr on an A10G HF Job. |
| Vector store | Qdrant **local mode** at `data/qdrant/` (matches Spaces-Pro persistent-disk layout). Two collections: `jobs_dense` (named dense + sparse vectors, HNSW + int8 scalar quantization on dense) and `jobs_multivec` (ColBERT MaxSim, populated by v1.1). |
| Pipeline | dense first-pass (top 100) → optional sparse search → RRF fusion (k=60) → optional cross-encoder rerank (`bge-reranker-v2-m3` or lite ms-marco MiniLM) → parent-chunk hydration → top-K. Filters on country, seniority_label_v1, role_family_v1, predicted_salary_usd_v1 range, posted_at. |
| Index time | 14:05 wall-clock for the full MiniLM index on Apple MPS (142 chunks/sec). |
| Latency | <1 s end-to-end on the matcher tab without the cross-encoder reranker; ~3-5 s with rerank enabled. |

The Matcher tab lives next to Salary Prediction and Search on the
Gradio Space — paste a query or resume blurb, apply filters, get
ranked jobs with predicted salary, classifier-derived seniority + role
family, top skills, and a snippet from the contributing parent chunk.

To re-index locally:

```sh
# dev / fast iteration (MiniLM, ~14 min for 120k chunks)
uv run python -m scripts.index_jobs --lite --force-recreate

# production (bge-m3 dense + sparse, ~6-8 hr on MPS)
uv run python -m scripts.index_jobs --force-recreate
```

### Retrieval eval — multi-variant comparison

Eval set: **48 labeled queries** (30 title-as-query with `(normalized
title, country)`-pool gold; 18 hand-crafted role+seniority queries with
classifier-label-pool gold). Run on the live MiniLM index over 12,334
jobs / 120k child chunks.

| Variant | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 | latency |
|---|---|---|---|---|---|---|
| `dense` | 0.291 | 0.363 | 0.393 | 0.412 | 0.349 | 186 ms/q |
| `hybrid+rerank` | **0.421** | **0.486** | **0.511** | **0.518** | **0.476** | 700 ms/q |

Cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2` lite) lifts
recall@10 by **+34%** for ~4x the latency. The bge-m3 reindex (v1.1)
adds the sparse leg + ColBERT MaxSim reranking, both of which CLAUDE.md
§8 expects to push recall further. HyDE (Qwen2.5-7B hypothetical-doc
generation before retrieval) lands as a UI toggle in Phase 6 follow-up.

Reproduce: `uv run python -m eval.run_retrieval_eval --variants dense hybrid+rerank`.

---

## NL→SQL analytics — Phase 7

Phase 7 ships an Analytics tab that turns plain-English questions into
DuckDB SQL over the curated corpus. The mandatory safety layer
(CLAUDE.md §11) is the senior-DS-signal piece — it's not optional:

| Layer | What it does |
|---|---|
| **Pre-parse keyword filter** | Word-boundary scan rejects `INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/COPY/ATTACH/PRAGMA/INSTALL/LOAD/...` even before the parser runs. Catches keyword-in-string-literal injection conservatively (false positives accepted). |
| **sqlglot parse** | Must yield exactly one statement; non-`Select`/`Subquery` rejected. |
| **Table allowlist** | Only `jobs` is queryable; CTE-introduced names allowed via `WITH ...`. Disallowed tables in `JOIN` clauses are caught. |
| **Column allowlist** | Per-table allowlist of ~40 columns (raw curated + Phase 4 versioned predictions). SELECT-list aliases and CTE columns are admitted; everything else rejected. |
| **Row + time caps** | DuckDB `statement_timeout=5s` + outer `LIMIT 1000` wrap. |
| **Always-show SQL** | The executed SQL is rendered alongside results so the user can verify what actually ran. |

**LLM backends** (auto-selected): `AnthropicLLM` (Claude Sonnet, when
`ANTHROPIC_API_KEY` is set) → `HFInferenceLLM` (Qwen2.5-7B-Instruct, when
`HF_TOKEN` has Inference-Provider permission) → fail-loud. `MockLLM` is
used in tests so CI never depends on a live API.

**61 safety-layer tests** lock the contract: each denylisted keyword
triggers a rejection, multi-statement payloads are caught, every legal
SELECT shape (joins, CTEs, subqueries, aggregates with aliases used in
ORDER BY/HAVING) passes, every illegal shape is rejected. Run:
`uv run pytest tests/rag/test_nl2sql.py -q`.

Reproduce a query end-to-end (set `ANTHROPIC_API_KEY` or an
inference-permitted `HF_TOKEN`):

```sh
uv run python -m rag.nl2sql "median predicted salary by country for senior MLEs"
```

**Live numbers from a 4-query smoke test** (Qwen2.5-7B via HF Inference,
against the live `data/curated_enriched/jobs.parquet`):

| Question | LLM-generated SQL | Result |
|---|---|---|
| _How many senior MLE jobs are open in the US right now?_ | `WHERE seniority_label_v1='senior' AND role_family_v1='MLE' AND country='US'` | **276** |
| _Top 5 companies hiring data scientists, ranked by number of postings._ | `GROUP BY company_name ORDER BY n DESC LIMIT 5` | Pinterest 65, Robinhood 61, Databricks 47, Whatnot 45, Jane Street 23 |
| _What is the average disclosed salary range for staff-level roles?_ | `AVG((salary_min + salary_max)/2) WHERE salary_disclosed` | **$212,863** USD/yr |
| _Distribution of role_family_v1 across countries._ | `GROUP BY country, role_family_v1` | 12 rows; US dominates RS (1,238), DA (2,918), DE (3,492); CA at 28, 122, ... respectively |

LLM accuracy on this set: 4/4. (One earlier query asked for "senior MLE
median salary" and the LLM filtered on `principal` — small sample, so
treat the smoke test as illustrative rather than benchmarked. A proper
NL→SQL eval set lands in v1.1 alongside the drift dashboard.)

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
