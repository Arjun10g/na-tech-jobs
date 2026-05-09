---
license: mit
task_categories:
  - tabular-regression
  - text-classification
  - feature-extraction
language:
  - en
tags:
  - jobs
  - hiring
  - data-science
  - machine-learning
  - north-america
  - salary
size_categories:
  - 10K<n<100K
pretty_name: "North American Tech Jobs"
configs:
  - config_name: default
    data_files:
      - split: train
        path: "curated/jobs.parquet"
---

# na-tech-jobs

A working ML platform for the senior data-science / ML hiring market in
the US and Canada. Weekly ingest from public ATS APIs, four trained
models, hybrid retrieval, and an LLM analytics layer. Runs on a $9/month
Hugging Face Space.

I started building this for my own job search. Each piece exists because
I had a question I couldn't answer with LinkedIn or a spreadsheet: what's
the actual salary distribution for senior MLE roles in Toronto, which
companies are hiring at staff level right now, where do my skills line up
with what's open. The ingestion cron, the salary regressor, the
seniority and role-family classifiers, the matcher, the analytics tab —
each one is the answer to one of those questions.

## 🔗 Live links

| | |
|---|---|
| 🚀 **Demo Space** (dark theme, 5 tabs) | https://arjun10g-na-tech-jobs.hf.space |
| 📦 **Source code** | https://github.com/Arjun10g/na-tech-jobs |
| 📊 **Dataset** (12,334 active jobs, weekly) | https://huggingface.co/datasets/arjun10g/na-tech-jobs |
| 🧠 **Models on the Hub** | [salary](https://huggingface.co/arjun10g/na-tech-jobs-salary-v1) · [seniority](https://huggingface.co/arjun10g/na-tech-jobs-seniority-v1) · [role_family](https://huggingface.co/arjun10g/na-tech-jobs-role_family-v1) · [skills](https://huggingface.co/arjun10g/na-tech-jobs-skills-v1) |

## What's live on the Space

Five tabs, all hitting real data and real model output.

**Salary.** Paste a job description, the regex cascade extracts ~20
features, an XGBoost regressor predicts the maximum salary in USD/year.
Edit any extracted field and the prediction updates. Held-out test-MAE
is $29,091, MAPE 14.7%.

**Search.** Substring match on title and company, with country and
role-family dropdowns. It's the boring tab — useful for spot-checking
the corpus before reaching for the matcher.

**Matcher.** Natural-language query (or a pasted resume blurb) goes
through dense Qdrant retrieval, an optional cross-encoder rerank, and
parent-chunk hydration. You get a ranked table of jobs with a clickable
apply link, plus a short LLM-generated paragraph explaining which 1-2
jobs best fit and why. Recall@10 against a labeled query set is 0.486
with rerank, 0.363 without.

**Analytics.** Plain-English question → Qwen2.5-7B writes DuckDB SQL →
sqlglot-based safety layer rejects anything that isn't a read-only
SELECT over the allowlist → DuckDB executes against the enriched
parquet. The executed SQL is shown next to the result so you can verify
what actually ran. 61 tests pin the safety contract; 4 out of 4 sample
questions returned the right answer in a live smoke test.

**Dashboard.** Pipeline health (last ingest, last enrichment, per-extractor
counts), market-trend tables (salary by role × seniority, top employers,
role-family share by country, top skills), and the latest drift report.
Refresh button pulls the curated parquet from the Hub on first call.

---

## Architecture

```mermaid
flowchart TB
    subgraph Ingest["⏱ Weekly — Sundays 02:00 UTC"]
        A1[Greenhouse]
        A2[Lever]
        A3[Ashby]
        A4[Workable]
        A5[SmartRecruiters]
        A6[Workday tenants]
        A1 & A2 & A3 & A4 & A5 & A6 --> ORCH[orchestrator<br/>asyncio + httpx]
        ORCH --> NORM[normalize.py<br/>title/loc/currency/period]
        NORM --> DEDUP[dedup vs prior snapshot]
        DEDUP --> QUAL[Pandera schema validation]
        QUAL --> SNAP[(snapshots/&lt;date&gt;/jobs.parquet)]
    end

    SNAP --> CURATE[curated/build.py<br/>DuckDB]
    CURATE --> CURATED[(curated/jobs.parquet<br/>12,334 active jobs)]

    subgraph Features["📦 Phase 1b — feature extraction cascade"]
        CURATED --> RGX[Tier 1 regex extractors<br/>seniority, role, skills,<br/>salary, sponsorship, ...]
        RGX -. opt-in .-> NUE[Tier 2 NuExtract LLM<br/>monthly retrain only]
    end

    subgraph Models["🧠 Phase 2-4 — models"]
        CURATED --> XGB[XGBoost salary regressor<br/>test-MAE $29,091]
        CURATED --> SENMODEL[MiniLM + LR seniority<br/>val f1_macro 0.812]
        CURATED --> ROLEMODEL[MiniLM + LR role_family<br/>val f1_macro 0.934]
    end

    XGB & SENMODEL & ROLEMODEL --> ENRICH[curated/enrich.py<br/>versioned predictions]
    ENRICH --> ENRICHED[(curated_enriched/jobs.parquet<br/>seniority_label_v1, role_family_v1,<br/>predicted_salary_usd_v1, ...)]

    subgraph RAG["🔍 Phase 5-6 — retrieval"]
        ENRICHED --> CHUNK[parent-child chunking<br/>29k parents, 120k children]
        CHUNK --> EMB[MiniLM dense<br/>bge-m3 v1.1 reindex]
        EMB --> QDRANT[(Qdrant local-mode<br/>jobs_dense + jobs_multivec)]
        QDRANT --> RAGPIPE[hybrid pipeline<br/>RRF + cross-encoder rerank<br/>recall@10 = 0.486]
    end

    subgraph App["🛰 HF Space — always-on"]
        ENRICHED --> SALARYTAB[Salary tab]
        ENRICHED --> SEARCHTAB[Search tab]
        RAGPIPE --> MATCHERTAB[Matcher tab]
        ENRICHED --> ANALYTICSTAB[Analytics tab<br/>NL→SQL + sqlglot safety]
        ENRICHED --> DASHTAB[Dashboard tab<br/>drift + market trends]
    end

    subgraph Ops["🔁 Closed-loop ops"]
        SNAP & ENRICHED --> DRIFT[Mon 03:00 UTC drift cron<br/>Evidently PSI]
        DRIFT -. PSI ≥ 0.20 .-> RETRAIN
        DRIFT --> ALERT[Discord webhook]
        RETRAIN[1st @ 04:00 UTC monthly<br/>champion/challenger gate] --> XGB & SENMODEL & ROLEMODEL
        RETRAIN --> ALERT
    end

    classDef artifact fill:#fef3c7,stroke:#d97706,color:#000
    classDef live fill:#dbeafe,stroke:#2563eb,color:#000
    classDef ops fill:#fee2e2,stroke:#dc2626,color:#000
    class SNAP,CURATED,ENRICHED,QDRANT artifact
    class SALARYTAB,SEARCHTAB,MATCHERTAB,ANALYTICSTAB,DASHTAB live
    class DRIFT,RETRAIN,ALERT ops
```

It's one loop. Ingest pulls from ATS APIs every Sunday, the curated
layer dedups and validates, the enrichment script scores every job with
the four models and writes a parquet with versioned columns
(`predicted_salary_usd_v1`, `seniority_label_v1`, …). The chunker +
embedder + Qdrant index serve the matcher; the same enriched parquet
backs analytics and the dashboard. Drift detection runs every Monday;
the retrain cron runs on the 1st of each month with a champion/challenger
gate. Everything's versioned through git and HF Hub commits. Detailed
architecture detailed in the source repo.

---

## Headline numbers

| Surface | Metric | Value |
|---|---|---|
| **Dataset** | Active jobs in latest snapshot | 12,334 |
| | Companies | 477 |
| | Salary disclosure rate | 49.8% |
| | Median disclosed / predicted salary (USD/yr) | $195k / $187.5k |
| **Salary regressor** | Test-MAE (XGBoost + Optuna 50) | **$29,091** (95% CI $27k–$31k) |
| | Test-MAPE | 14.7% |
| | R² log-salary | 0.730 |
| **Seniority classifier** | f1_macro vs reviewed gold | **0.812** (95% CI [0.73, 0.87]) |
| **Role-family classifier** | f1_macro vs reviewed gold | **0.934** (95% CI [0.88, 0.98]) |
| **Hybrid retrieval** | recall@10 (`hybrid+rerank`, 48 labeled queries) | **0.486** (vs `dense` 0.363) |
| | MRR | 0.518 |
| **NL→SQL** | Live smoke set accuracy | 4/4 |
| | Safety-layer test count | 61 |
| **CI** | Total tests passing | **371** |

---

## Phase status

| Phase | Status | Headline |
|---|---|---|
| 0–1 — Scaffold + Ingestion | ✅ | Repo + CI + 65 ATS handles → weekly parquet → HF Dataset (~12.3k jobs) |
| 2 — Salary regressor | ✅ | Six-tier ladder; **Tier 5 XGBoost test-MAE $29,091**, every CI cleanly excludes the previous |
| 3 — First deployable | ✅ | Salary prediction + curated search live on the Space |
| 4 — Multi-model + payload enrichment | ✅ | MiniLM + LR classifiers (seniority val 0.812, role_family 0.934), regex skills, all 12,334 jobs enriched with versioned predictions |
| 5 — Retrieval stack | ✅ | Parent-child chunking (29k/120k) + Qdrant + dense (MiniLM) hybrid + cross-encoder rerank + Matcher tab; bge-m3 reindex queued v1.1 |
| 6a — Retrieval eval | ✅ | 48-query labeled set; `hybrid+rerank` recall@10 = **0.486** (+34% vs dense) |
| 7 — NL→SQL analytics | ✅ | Mandatory sqlglot safety layer + 61 tests; Anthropic / HF Inference / mock LLM backends; Analytics tab live |
| 8a — Dashboard | ✅ | Drift detection (PSI) + pipeline-health + market-trend tabs |
| 8b — CI workflows | ✅ | `drift.yml` (Mondays 03:00 UTC) + `retrain.yml` (monthly, champion/challenger gate) |
| 9 — Polish | ✅ | Mermaid architecture diagram, dark UI, HF dataset YAML frontmatter, model-card cross-links |

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

Each tier's 95% bootstrap CI sits cleanly above the previous tier's;
test-MAE and CV-MAE agree to within 5%, so this isn't a lucky test
draw or an overfit. The original target of MAE under $25k isn't met
yet; closing that gap is the bge-m3 description-embedding work in v1.1.

Methodology is parsimonious-first, informed by Breiman's two-cultures
essay, the Mincer earnings function, and the recent gradient-boosting-
on-tabular results from Shwartz-Ziv & Armon (2022) and Grinsztajn et
al (2022).

---

## Title classifiers — frozen MiniLM + multinomial LR

| Classifier | Classes | Train rows | f1_macro vs regex | f1_macro vs **reviewed gold** | 95% CI | Repo |
|---|---|---|---|---|---|---|
| seniority | 7 (intern…director) | 6,361 | 0.831 | **0.812** | [0.73, 0.87] | [`...-seniority-v1`](https://huggingface.co/arjun10g/na-tech-jobs-seniority-v1) |
| role_family | 6 (DS / DA / DE / MLE / RS / AS / SWE-ML) | 569 | 0.915 | **0.934** | [0.88, 0.98] | [`...-role_family-v1`](https://huggingface.co/arjun10g/na-tech-jobs-role_family-v1) |

Architecture is frozen `sentence-transformers/all-MiniLM-L6-v2`
(22M params, 384-dim) for embeddings, sklearn multinomial LR for the
head. Class weights balanced. C picked by 5-fold stratified CV from
`{0.1, 1, 10}`.

Why a linear probe and not the originally-planned DeBERTa-v3 + LoRA?
For short-text classification on weakly-supervised labels, a linear
probe over a strong general-purpose embedder lands at the same operating
point for roughly two orders of magnitude less compute. The literature
that motivated the call is Peters et al (2019) on when fine-tuning
helps, Tunstall et al's SetFit, and Joulin's FastText. v1.2 will run
the DeBERTa-v3 comparison against human-reviewed gold once that test
set lands.

Training labels come from the regex extractors. Rows where the regex
fell back to its default (`mid` / `Other` / `Manager`) are dropped from
training because the fallback is too noisy. The "vs regex" column
measures held-out agreement against the same regex labels; "vs reviewed
gold" comes from a two-pass Claude-reviewed sample (230 rows per
classifier, first-pass labelers then second-pass reviewers shown the
proposal + the classifier's prediction). The reviewer override rate
was 1.7%.

Skills are regex-first by default. `extracted_skills_v1` gets populated
from [`ingestion/feature_extraction/regex/tech_stack.py`](ingestion/feature_extraction/regex/tech_stack.py)
on every weekly ingest — about 70 canonical names, 64.7% coverage,
runs in milliseconds, and (most importantly) is free. NuExtract
(`arjun10g/na-tech-jobs-skills-v1`) is wired but opt-in via
`--skills-mode=nuextract`. It runs during the monthly retrain so the
enriched skills column gets the LLM-tier output once a month.

`curated_enriched/jobs.parquet` on the Hub has all 12,334 jobs scored
with the versioned columns: `seniority_label_v1`,
`seniority_confidence_v1`, `role_family_v1`, `role_family_confidence_v1`,
`predicted_salary_usd_v1`, `extracted_skills_v1`,
`prediction_model_version`.

```sh
uv run python -m curated.enrich --push-to-hub  # rebuild + push
```

---

## Hybrid retrieval (Phase 5)

| | Detail |
|---|---|
| Chunking | parent-child `RecursiveCharacterTextSplitter` (~1024-tok parents, ~256-tok / 32-overlap children, hierarchical markdown separators) |
| Volume indexed | **12,334 jobs → 29,311 parents → 120,004 children** |
| Embedder (v1) | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, dense-only). bge-m3 (1024-dim dense + sparse + ColBERT multivec) wired but reindex deferred to v1.1 |
| Vector store | Qdrant local-mode at `data/qdrant/`. Two collections: `jobs_dense` (named dense + sparse vectors, HNSW + int8 scalar quantization) and `jobs_multivec` (ColBERT MaxSim, populated by v1.1) |
| Pipeline | dense first-pass (top 100) → optional sparse search → RRF fusion (k=60) → optional cross-encoder rerank (`bge-reranker-v2-m3` or lite ms-marco MiniLM) → parent-chunk hydration → top-K → **LLM rationale** |
| Index time | 14:05 wall-clock for the full MiniLM index on Apple MPS (142 chunks/sec) |
| Latency | <1 s end-to-end without rerank; ~3-5 s with rerank enabled |

A note on how the index gets to the Space: `data/` is gitignored and
excluded from the deploy. The Qdrant directory ships out-of-band as a
gzipped tarball at `arjun10g/na-tech-jobs/qdrant/qdrant_minilm_v1.tar.gz`
(~260 MB). On the Space's first matcher request,
`app/retriever_loader.py` downloads and extracts it. Subsequent requests
hit the local directory. The same pattern will work for the bge-m3
tarball when it lands.

### Retrieval eval

48 labeled queries: 30 are sampled job titles with the gold pool defined
by all jobs sharing the same normalized title and country. The other 18
are hand-written role+seniority queries with a gold pool defined by the
classifier-label slice on the enriched parquet.

| Variant | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 | latency |
|---|---|---|---|---|---|---|
| `dense` | 0.291 | 0.363 | 0.393 | 0.412 | 0.349 | 186 ms/q |
| `hybrid+rerank` | **0.421** | **0.486** | **0.511** | **0.518** | **0.476** | 700 ms/q |

The cross-encoder is the biggest single quality lever in v1: +34% on
recall@10 for ~4x the latency. Hybrid here is identical to dense
because the MiniLM index has no sparse vectors yet; the bge-m3 reindex
adds the sparse leg, and the same indexer already wires the ColBERT
multi-vec collection. HyDE (a Qwen-generated hypothetical doc fed to
retrieval before the real query) is queued behind the bge-m3 reindex
since both rows benefit from sparse search.

```sh
# dev / fast iteration (MiniLM, ~14 min for 120k chunks)
uv run python -m scripts.index_jobs --lite --force-recreate
# production (bge-m3 dense + sparse, ~6-8 hr on MPS or ~30 min on A10G HF Job)
uv run python -m scripts.index_jobs --force-recreate

# eval
uv run python -m eval.run_retrieval_eval --variants dense hybrid+rerank
```

---

## NL→SQL analytics

The Analytics tab takes a plain-English question and runs DuckDB SQL
over the enriched parquet. The interesting work isn't in the LLM call,
it's in the safety layer that sits between the LLM and the database.
Six things have to pass before the SQL is allowed to run:

| Layer | What it does |
|---|---|
| **Pre-parse keyword filter** | Word-boundary scan rejects `INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/COPY/ATTACH/PRAGMA/INSTALL/LOAD/...` *before* the parser runs. Conservatively catches keyword-in-string-literal attacks. |
| **sqlglot single-statement parse** | One statement, must be a `Select` or `Subquery`. |
| **Table allowlist** | Only `jobs` is queryable; CTE-introduced names admitted. Disallowed tables in `JOIN` clauses caught. |
| **Column allowlist** | Per-table allowlist of ~40 columns. SELECT-list aliases and CTE columns admitted; everything else rejected. |
| **Row + time caps** | DuckDB `statement_timeout=5s` + outer `LIMIT 1000`. |
| **Always-show SQL** | Executed SQL rendered alongside results so the user verifies what actually ran. |

`default_llm()` picks a backend automatically: Anthropic Claude when
`ANTHROPIC_API_KEY` is set, HF Inference / Qwen2.5-7B-Instruct when only
`HF_TOKEN` is, fail-loud otherwise. Tests use a `MockLLM` so CI never
depends on a live API.

61 safety-layer tests lock the contract. Every denylisted keyword is
parametrized, multi-statement payloads get caught, every legal SELECT
shape (joins, CTEs, subqueries, aggregates with aliases used in ORDER
BY / HAVING) passes, every illegal shape gets rejected.

Live smoke test against Qwen2.5-7B over the enriched parquet:

| Question | Result |
|---|---|
| _How many senior MLE jobs are open in the US right now?_ | **276** |
| _Top 5 companies hiring data scientists, ranked by number of postings._ | Pinterest 65, Robinhood 61, Databricks 47, Whatnot 45, Jane Street 23 |
| _Average disclosed salary range for staff-level roles?_ | **$212,863** USD/yr |
| _Distribution of role_family_v1 across countries._ | 12 rows; US dominates DE (3,492), DA (2,918), SWE-ML (2,473); CA SWE-ML-heavy at 29.6% |

```sh
uv run python -m rag.nl2sql "median predicted salary by country for senior MLEs"
```

---

## Operations

Four crons close the loop. Discord webhooks fire on success and failure.

| Cron | When | What |
|---|---|---|
| [`ingest.yml`](.github/workflows/ingest.yml) | Sun 02:00 UTC | Pull every ATS, dedup vs prior, validate, push snapshot to dataset repo, Discord alert on failure |
| [`drift.yml`](.github/workflows/drift.yml) | Mon 03:00 UTC | Compare latest vs 4-week-old snapshot via Evidently. PSI ≥ 0.20 on any tracked feature → priority breach → flag retrain + Discord alert |
| [`retrain.yml`](.github/workflows/retrain.yml) | 1st @ 04:00 UTC | Matrix over `[seniority, role_family]`. Pull champion `training_summary.json`, train challenger, apply [`monitoring.champion_challenger`](monitoring/champion_challenger.py) gate (primary +1%, no secondary > -2%), publish only if promoted |
| [`deploy-space.yml`](.github/workflows/deploy-space.yml) | On push to main | Generate `requirements.txt` from `space-runtime` extras, rsync runtime files, push to HF Space, write Gradio frontmatter README |

The dashboard tab pulls whichever drift report is newest in
`reports/drift/<date>.html` and renders it inline next to a metrics
card. The pipeline-health card reads `ingestion_stats.json` from the
latest snapshot directory.

---

## Data dictionary

Every column in `data/curated/jobs.parquet` (type, fill rate, whether
it's a predictor) is documented in
[`DATA_DICTIONARY.md`](DATA_DICTIONARY.md). The enriched parquet adds
the seven versioned-prediction columns described in the
[Title classifiers](#title-classifiers--frozen-minilm--multinomial-lr)
section above.

---

## Quickstart

```sh
# clone + install Python 3.11 + dev deps
uv sync --group dev

# run the local Gradio app (loads model + curated parquet from HF Hub)
uv run python -m app.main

# run lint + tests
uv run ruff format --check .
uv run ruff check .
uv run pytest

# smoke ingest (5 companies, no HF push)
uv run python -m ingestion.orchestrator --output-dir data --limit 5

# full ingest with HF Dataset push (needs HF_TOKEN)
uv run python -m ingestion.orchestrator --output-dir data --push-to-hub --alert

# rebuild curated layer + enriched predictions
uv run python -m curated.build --push-to-hub
uv run python -m curated.enrich --push-to-hub

# retrain a classifier + push to HF Hub
uv sync --extra ml --group dev
uv run python -m models.seniority.train  # ~30 sec on CPU
uv run python -m scripts.publish_classifier seniority --create

# rebuild Qdrant index + publish tarball
uv run python -m scripts.index_jobs --lite --force-recreate
tar -cf - -C data qdrant | gzip -9 > qdrant_minilm_v1.tar.gz
uv run python -m scripts.publish_qdrant_index --tarball qdrant_minilm_v1.tar.gz
```

### Optional dependency groups

| Group | Purpose | Install |
|---|---|---|
| `ml` | Training stack (torch, transformers, xgboost, mlflow, sklearn, optuna) | `uv sync --extra ml` |
| `eda` | matplotlib, seaborn, statsmodels, missingno, tabulate | `uv sync --extra eda` |
| `space-runtime` | Lean runtime deps for the Space (xgboost, sklearn, sqlglot, qdrant-client, sentence-transformers, anthropic) | `uv sync --extra space-runtime` |
| `rag` | Vector store, chunking, NL→SQL, PDF parsing | `uv sync --extra rag` |
| `monitoring` | Evidently for drift reports | `uv sync --extra monitoring` |
| `api` | FastAPI for programmatic endpoints | `uv sync --extra api` |

## Repository layout

```
app/                  Gradio front end (5 tabs, dark-themed)
ingestion/            ATS extractors + feature cascade
curated/              DuckDB layer over weekly snapshots, model-prediction enrichment
eda/                  Statistical audit + train/test split
models/               salary, seniority, role_family, skills, embeddings
rag/                  Chunking, embedder, qdrant client, reranker, pipeline, nl2sql
monitoring/           Drift, pipeline health, market trends, champion/challenger gate
eval/                 Retrieval-eval harness, labeled queries, classifier test sets
scripts/              Publish models/dataset, build query sets, index_jobs, etc.
.github/workflows/    CI, weekly ingest, drift cron, monthly retrain, deploy-space
tests/                371 tests across all surfaces
```

Full structure on [GitHub](https://github.com/Arjun10g/na-tech-jobs).

## Secrets

Copy `.env.example` to `.env` and fill in `HF_TOKEN` (with both Hub-write
**and** Inference-Provider permissions if you want the Analytics tab to use
Qwen) and `DISCORD_WEBHOOK_URL` for ingest/drift/retrain alerts.
`ANTHROPIC_API_KEY` is optional and preferred over HF Inference when set.

## License

MIT. Models and datasets licensed individually — see their respective cards.
