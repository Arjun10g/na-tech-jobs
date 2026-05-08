# CLAUDE.md — North American Tech Jobs ML Platform

> Working name: `na-tech-jobs` (final name TBD by owner). This document is the project bible — every architectural decision, model choice, phase of work, and known tradeoff lives here. When in doubt, this file wins.

## Decision deltas

Living deltas where reality diverged from the original plan. Newest first.

- **2026-05-08 — NuExtract Tier 2 wired but DORMANT.** Step 1b finished the cascade end-to-end: real NuExtract-tiny wrapper, batched generation, MPS+float16, schema/coercion/enum validation, mock-based tests. **Default `LLM_ELIGIBLE_FIELDS = frozenset()` — Tier 2 doesn't run.** Why: benchmarks landed at 3.6 s/row on M-series MPS (12.3 hrs for 12k rows) for uplift on fields (`industry_experience`, `team_or_department`, `tech_stack`-prose, `requires_citizenship`, `offers_visa_sponsorship`) that no current downstream consumer uses, and the bge-m3 description embedding in Phase 5 carries the same semantic signal anyway. Re-enable by populating `LLM_ELIGIBLE_FIELDS` and either running locally overnight, on HF Jobs / ZeroGPU, or via a hosted LLM. All wiring + tests stay in place.
- **2026-05-08 — Feature-extraction cascade pulled forward to Phase 2.** Originally NuExtract was scoped to Phase 4 only. We now build a regex-first cascade in Phase 2 Step 1a (regex Tier 1 only) → Step 1b wires NuExtract-tiny as Tier 2. Rationale: Phase 1's read-through revealed ~20 high-value features (security clearance, citizenship, sponsorship, equity form, bonus type, contract type, posting quality, tech stack, etc.) buried in description text, *all* of which improve the salary regressor. Building the cascade once, with both tiers, beats two passes. Phase 4 still owns the seniority + role-family DeBERTa classifiers. See §7 task list and `ingestion/feature_extraction/` for the new module layout.
- **2026-05-08 — Python 3.11 floor relaxed to 3.10.** HF Spaces' Gradio SDK image hard-codes Python 3.10 and ignores the `python_version` frontmatter. We resolve dependencies for 3.10 in CI / Space pushes; local dev + GH Actions stay on 3.11 via `.python-version`. See `infra/secrets.md` and the deploy workflow for the resolution flow.

---

## 1. Project mission

Build a production-grade, fully open-source ML platform for the North American senior tech-hiring market: weekly ATS ingestion across Greenhouse, Lever, Ashby, Workable, SmartRecruiters, and curated Workday tenants → a curated dataset on the Hugging Face Hub → a multi-model pipeline (salary regression, seniority and role-family classifiers, skill extraction, embeddings) → a hybrid + late-interaction RAG layer with LLM-powered retrieval and natural-language analytics → all deployed on HF Spaces Pro with ZeroGPU, with weekly drift monitoring and monthly automated retraining.

The project's seed user is the builder — a senior data science candidate using the platform on their own North American job search. That's the point: every component should be useful to someone actually job-hunting, not a contrived demo.

---

## 2. Why this project exists (the senior DS context)

This project is engineered to demonstrate every signal that came up across senior data science postings the builder analyzed (Citi, Tripstack/Etraveli, AGCO, Pelmorex, plus the broader NA market). Those postings repeatedly emphasized:

- Python + SQL as the floor, not a differentiator.
- Classical ML stack (sklearn, XGBoost) plus deep learning (PyTorch, transformers).
- Cloud + distributed computing as baseline.
- **MLOps and full lifecycle ownership as the real gatekeeper** — concept → data prep → training → deployment → monitoring → retraining.
- Statistics and experimentation rigor (A/B testing, hypothesis testing, experimental design).
- Communication and cross-functional partnership in every single posting.
- GenAI / LLMs / agentic systems as the new differentiator at the senior end (Citi was almost entirely this).
- Domain context — increasingly explicit in postings.

This project's components map directly to those signals. Every architectural decision below was made with that mapping in mind. When tradeoffs come up during build, prefer the option that demonstrates the senior-DS signal more clearly, even at modest cost in implementation simplicity.

The narrative arc, ready to use in interviews and the README:

> "I built a production ML platform for the North American senior tech-hiring market — weekly ATS ingestion across Greenhouse, Lever, Ashby, and Workday, with salary, seniority, and role classifiers feeding a hybrid + late-interaction RAG layer over 100k+ live job descriptions. Drift monitoring, champion/challenger model promotion, and automated retraining run on a $9 Hugging Face Pro plan. I used it for my own job search."

That sentence is the goal. Build toward it.

---

## 3. Locked decisions

The following decisions are locked. Do not re-litigate during build unless evidence emerges that one is materially wrong.

1. **Geographic scope: North America only.** US + Canada. EU is out of scope for v1.
2. **Workday is in.** Despite extractor complexity, Workday-using enterprises concentrate the senior DS roles we care about. Plan for a smaller initial Workday tenant list (30–50 enterprises) that grows over time, with a per-tenant config.
3. **Aramente/eu-tech-jobs is dropped.** Not used as data source, not used as comparison set. Cited in README as methodological reference only.
4. **English-only for MVP.** Quebec French postings are filtered out at ingest. French handling is a v2 stretch goal.
5. **Retraining cadence: monthly full pipeline, weekly drift detection.** Drift checks run every Monday after ingest; full model retraining is triggered on the first of each month or by drift exceeding thresholds.
6. **Always-on Spaces deployment.** $9 Pro tier with always-on enabled, accepting the compute-hour cost in exchange for recruiter-friendly UX.
7. **Fully open-source stack.** All models from HuggingFace, all code under MIT or Apache-2.0 licenses, all data under permissive license with company-removal mechanism.
8. **Single-repo monorepo.** Ingestion, models, RAG, app, eval all live in one repo. No premature service splitting.
9. **No LinkedIn / Indeed scraping.** ToS-prohibited. ATS APIs are the source of truth for actual applications.

---

## 4. Architecture overview

The system is a single flywheel. Components are designed so that data, models, and predictions reinforce each other rather than living in silos.

### Data flow

1. **Ingestion** (weekly, GitHub Actions cron, Sundays 02:00 UTC): per-provider extractors pull jobs from public ATS APIs and Workday tenant endpoints, normalize to a single schema, deduplicate against prior snapshots, write parquet to the HF Dataset repo as a versioned commit.
2. **Quality gates** (in the same Action): Pandera schema validation, plausibility checks, currency normalization. Failures halt the pipeline and post to Discord webhook.
3. **Curated layer** (DuckDB queries against the parquet snapshots): a normalized, deduplicated, salary-normalized rolling table with derived features. Pushed to HF Dataset under a separate path/branch.
4. **Model inference enrichment** (monthly, after retraining): every active job in the curated table gets scored by every model. Predictions become payload fields. Versioned (e.g., `predicted_salary_usd_v3`) so prior versions remain available.
5. **Vector indexing** (after enrichment): bge-m3 produces dense, sparse, and (on-demand) multi-vector embeddings. Dense + sparse for all chunks goes into Qdrant local mode with HNSW + int8 scalar quantization. Multi-vector ColBERT embeddings are computed on-the-fly for top-K candidates only.
6. **Serving** (HF Spaces, Gradio app, ZeroGPU): user queries flow through HyDE (optional) → hybrid retrieval → cross-encoder reranking → optional ColBERT late interaction → context summarization → final LLM generation. Predictions in payload allow filtering at retrieval time.
7. **Drift detection** (weekly, GitHub Actions): Evidently AI reports compare the new snapshot to a 4-week rolling baseline. Reports written to `reports/` in Dataset repo. Threshold breaches trigger Discord webhook and flag the next monthly retraining as priority.
8. **Eval harness** (gates everything): a hand-labeled set of 50–100 query / relevance pairs, seeded from the builder's own resume and target queries, used for retrieval comparison, model promotion (champion/challenger), and post-drift validation.

### Key integration points

- **The same bge-m3 model produces all embeddings** — dense for retrieval, sparse for hybrid lexical search, ColBERT-style multi-vector for late-interaction reranking, AND the dense vectors are used as features in the salary regressor. One model, four roles.
- **The same Qwen2.5-7B model handles all LLM tasks** — HyDE generation, context summarization, NL→SQL translation, and skill extraction (where NuExtract isn't sufficient). Loaded once, used for everything.
- **Predictions enrich retrieval, retrieval informs the user.** Predicted salary, seniority, and role become Qdrant payload fields, enabling queries the raw data alone can't answer ("senior MLE roles in Toronto predicted to pay above $180k CAD").
- **Drift closes the loop.** Weekly drift detection flags shifts → triggers next monthly retraining → updated predictions → re-enriched payloads → updated retrieval results.

---

## 5. Tech stack

### Languages and runtime
- **Python 3.11+** — exclusive language across ingestion, models, RAG, app.
- **uv** for environment / dependency management (faster than pip+venv, simpler than poetry).

### Storage and compute
- **HuggingFace Hub** — single backbone for storage of versioned datasets, model checkpoints, and the deployed Space.
- **HuggingFace Spaces Pro ($9/month)** with **always-on enabled** and **persistent storage** (~20GB allocated) for the deployed app and Qdrant index.
- **ZeroGPU** for inference — function-decorated GPU calls (`@spaces.GPU` with explicit duration). Models live on CPU at Space startup, move to GPU per-request.
- **GitHub Actions** for orchestration — weekly ingest cron, weekly drift cron, monthly retraining cron, CI for tests and lint.
- **No external cloud (AWS/GCP/Azure)** — keeps the project free-tier and simple.

### Data layer
- **parquet** for all snapshots and curated tables.
- **DuckDB** as the query engine — reads parquet directly, full SQL, scales to multi-million rows on a laptop.
- **Pandera** for data validation (lighter than Great Expectations, fits in CI).

### ML layer
- **scikit-learn** + **XGBoost** for tabular models (salary regression, where appropriate).
- **transformers** + **PEFT** (LoRA) for encoder fine-tuning (seniority and role-family classifiers).
- **sentence-transformers** + **FlagEmbedding** for bge-m3 and reranker.
- **MLflow** for experiment tracking and model registry. Backed by SQLite + local artifact store (lightweight) or pushed to HF Models.

### RAG layer
- **bge-m3** (`BAAI/bge-m3`) — unified embedder producing dense, sparse, and multi-vector outputs. The single most important model choice in the project.
- **bge-reranker-v2-m3** (`BAAI/bge-reranker-v2-m3`) — cross-encoder reranker, multilingual, matches bge-m3.
- **Qwen2.5-7B-Instruct** (`Qwen/Qwen2.5-7B-Instruct`) — LLM for HyDE, summarization, NL→SQL. Apache-2.0.
- **NuExtract-tiny-v1.5** (`numind/NuExtract-tiny-v1.5`) — purpose-built structured extraction for skills/tech stack.
- **Qdrant** (`qdrant-client[fastembed]`) — vector store, **local mode** with persistent storage on Spaces. HNSW indexing + int8 scalar quantization for the dense vectors.
- **LangChain** — recursive chunker only. No agents, no chains, no LangGraph. We use it for `RecursiveCharacterTextSplitter` and parent-child indexing patterns; everything else is hand-written.

### App layer
- **Gradio** for the user-facing UI — built-in SSO with Spaces, native Python, fits the audience (it's a portfolio piece, not a product).
- **FastAPI** for any programmatic endpoints (NL→SQL, /predict/*).

### Eval and monitoring
- **Evidently AI** for drift reports (weekly cron output).
- **Hand-rolled eval harness** — Python module computing recall@k, MRR, nDCG over the labeled query set. No third-party RAG eval framework — keep it ours.

### CI / dev tooling
- **ruff** for lint + format.
- **pytest** for tests.
- **pre-commit** hooks for ruff.
- **GitHub Actions** for CI on every PR.

---

## 6. Data model

### Canonical job schema

All extractors normalize to this schema. Stored in parquet, accessed via DuckDB.

| Column | Type | Notes |
|---|---|---|
| `id` | string | `sha256(company_slug + url)[:16]`, stable across snapshots |
| `company_slug` | string | matches an entry in `companies.yaml` |
| `company_name` | string | display name |
| `title` | string | as posted |
| `url` | string | canonical apply URL |
| `location_raw` | string | as reported by ATS |
| `country` | string | ISO 3166-1 alpha-2, normalized (`US`, `CA`) |
| `region` | string | state/province code where derivable |
| `city` | string | normalized; may be null |
| `remote_policy` | enum | `onsite` / `hybrid` / `remote` / `remote-na` |
| `seniority_extracted` | string | rule-based extraction from title (raw label, noisy) |
| `role_family_extracted` | string | rule-based, noisy |
| `salary_min`, `salary_max` | float | when disclosed |
| `salary_currency` | string | ISO 4217 (`USD`, `CAD`) |
| `salary_period` | enum | `year` / `month` / `day` / `hour` |
| `salary_min_usd_yearly`, `salary_max_usd_yearly` | float | normalized to USD/year for modeling |
| `salary_disclosed` | bool | derived |
| `description_md` | string | sanitized markdown |
| `posted_at` | timestamp | UTC, as reported |
| `scraped_at` | timestamp | UTC, ingest time |
| `source` | string | extractor name (`greenhouse`, `lever`, `ashby`, `workable`, `smartrecruiters`, `workday`) |
| `raw_payload_hash` | string | sha256 of raw extractor response, for change detection |

### Predictions (added by enrichment, versioned)

| Column | Type | Notes |
|---|---|---|
| `predicted_salary_usd_v{N}` | float | XGBoost output, USD/year |
| `seniority_label_v{N}` | enum | `intern` / `junior` / `mid` / `senior` / `staff` / `principal` / `manager` / `director` / `exec` |
| `seniority_confidence_v{N}` | float | softmax max |
| `role_family_v{N}` | enum | `DS` / `DA` / `DE` / `MLE` / `RS` / `AS` / `SWE-ML` / `Manager` / `Other` |
| `extracted_skills_v{N}` | list[string] | from NuExtract |
| `prediction_model_version` | string | which model versions produced these |

### Storage layout on HF Dataset repo

```
datasets/<owner>/na-tech-jobs/
├── README.md
├── companies/companies.yaml          # registry, hand-curated
├── snapshots/
│   ├── 2026-05-04/jobs.parquet       # weekly raw snapshot
│   ├── 2026-05-11/jobs.parquet
│   └── …
├── latest/jobs.parquet               # symlink/copy of most recent snapshot
├── latest/companies.parquet          # parsed yaml, for queryability
├── curated/jobs.parquet              # deduplicated, normalized, enriched (active jobs only)
├── curated/jobs_history.parquet      # all jobs ever seen, with last_seen_at
├── reports/drift/2026-05-11.html     # weekly Evidently reports
├── reports/quality/2026-05-11.json   # Pandera failure summaries
└── eval/queries.jsonl                # labeled eval set (committed for reproducibility)
```

Every weekly run is a single git commit. Roll back is `git revert`.

---

## 7. ML pipeline

### Models to build (in priority order)

1. **Salary regressor** (`models/salary/`)
   - **Target:** `salary_max_usd_yearly` (use max, not midpoint — disclosed ranges are typically lowballed at the min).
   - **Features:** tabular (country, region, seniority_extracted, role_family_extracted, source, remote_policy, has_equity, posted_month) + bge-m3 dense embedding of `description_md` (1024-dim), concatenated.
   - **Algorithm:** XGBoost regressor (sklearn API). Hyperparameter search via Optuna with 5-fold CV.
   - **Training data:** rows where `salary_disclosed = True`. Expect ~50–70% of NA postings.
   - **Eval:** held-out set stratified by source and country. Report MAE, MAPE, R² per stratum (don't just report aggregate — bias matters).
   - **Honest framing:** the model predicts "salary the way disclosing companies in our corpus would price this role." Document this in the model card.

2. **Seniority classifier** (`models/seniority/`)
   - **Labels:** 9-class `intern / junior / mid / senior / staff / principal / manager / director / exec`.
   - **Approach:** fine-tune `microsoft/deberta-v3-base` with LoRA on title + first 512 tokens of description.
   - **Training data:** weakly supervised initial labels from rule-based extraction over titles, then **hand-labeled clean test set of 500 examples** (this is the must-do; weak labels alone give a noisy classifier).
   - **Eval:** macro-F1 on hand-labeled test set, confusion matrix in model card.

3. **Role-family classifier** (`models/role_family/`)
   - **Labels:** 9-class as above (`DS` / `DA` / `DE` / `MLE` / `RS` / `AS` / `SWE-ML` / `Manager` / `Other`).
   - **Approach:** same as seniority (DeBERTa-v3-base + LoRA).
   - **Training data:** weakly supervised + 500-example hand-labeled test set.

4. **Skill extractor** (`models/skills/`)
   - **Approach:** zero-shot extraction via `numind/NuExtract-tiny-v1.5` with a defined output schema (programming languages, frameworks, cloud platforms, databases, ML libraries).
   - **No training.** Just inference + post-processing into a canonical taxonomy (see `taxonomies/skills.yaml`).
   - **Eval:** precision/recall on a hand-labeled 100-job set.

5. **Description embedder** (`models/embeddings/`)
   - Not "trained" — uses `BAAI/bge-m3` directly. Wraps loading + encoding logic.

### Training cadence

- **Monthly full retraining** on the 1st of each month: pull latest curated table, retrain all four supervised models (salary, seniority, role family, skills baseline), evaluate via champion/challenger against held-out test sets, promote winners to the production registry.
- **Weekly drift checks** (Mondays after ingest): run Evidently. If PSI > 0.2 on any tracked feature, flag next retraining as `priority`.
- **Champion/challenger gating:** new model promoted only if it beats production by ≥1% on the primary metric AND doesn't regress >2% on any secondary metric. Otherwise hold.

### MLflow setup

- Local tracking server backed by SQLite, artifacts stored in `mlruns/`.
- Tracking URI configurable via env var so CI runs can ship to a separate registry.
- Each model gets its own experiment. Metrics, params, and the trained artifact go to MLflow; the production winner gets a copy uploaded to its HF Model repo.
- Model cards (`README.md` in each Model repo) auto-generated from MLflow run metadata.

---

## 8. RAG pipeline

This is the most-detailed component because it's where the project demonstrates senior-level GenAI work.

### Chunking

- **`langchain.text_splitter.RecursiveCharacterTextSplitter`** with hierarchical separators: `["\n## ", "\n### ", "\n\n", "\n", ". ", " "]`.
- **Parent–child indexing.** Two parallel chunkings:
  - Child chunks: ~256 tokens, 32-token overlap. Used for retrieval (precision).
  - Parent chunks: ~1024 tokens, no overlap. Returned to the LLM (context completeness).
- **Metadata attached to every chunk:** `job_id`, `company_slug`, `country`, `region`, `role_family`, `seniority_label`, `posted_at`, `salary_min_usd_yearly`, `salary_max_usd_yearly`, `predicted_salary_usd`, `source`, `chunk_index`, `parent_chunk_id`.

### Embedding

- **`BAAI/bge-m3`** for everything.
- bge-m3 produces three outputs from one forward pass:
  - **Dense** (1024-dim float, mean-pooled) — for first-pass retrieval.
  - **Sparse** (lexical token weights) — for hybrid lexical search.
  - **Multi-vector** (ColBERT-style, ~128-dim per token, no pooling) — for late-interaction reranking.
- All three are computed at indexing time for child chunks. Multi-vector embeddings are stored compressed (int8 PQ) in a separate Qdrant collection because they're large.
- **Storage budget:** dense (240k chunks × 1024 × int8 ≈ 250MB), sparse (negligible, sparse), multi-vector (~3GB compressed). Total well under persistent storage budget.

### Vector store

- **Qdrant** in **local mode** with persistent storage at `/data/qdrant`.
- **Two collections:**
  - `jobs_dense` — dense vectors + payload, HNSW (m=16, ef_construct=200), int8 scalar quantization.
  - `jobs_multivec` — multi-vector for ColBERT, int8 PQ.
- **Sparse vectors** stored alongside dense in `jobs_dense` (Qdrant supports both natively since v1.7).
- Filters at query time use payload fields (`country`, `seniority_label`, `predicted_salary_usd_v3`, `posted_at`).

### Retrieval stack

The full retrieval path, in order:

1. **Optional HyDE** (toggle in UI; default off for speed):
   - User query → Qwen2.5-7B generates a 200-word hypothetical job posting that would answer the query.
   - Embed the hypothetical with bge-m3 dense.
   - Use that embedding for first-pass retrieval (instead of the raw query embedding).

2. **First-pass hybrid retrieval** → top 100 child chunks:
   - Dense search over `jobs_dense` (cosine).
   - Sparse search over `jobs_dense` (BM25-style on bge-m3 sparse weights).
   - Fuse with **Reciprocal Rank Fusion** (RRF, k=60).
   - Apply payload filters (country, seniority, salary band, etc.).

3. **Cross-encoder reranking** with `BAAI/bge-reranker-v2-m3` → top 20:
   - Score each (query, chunk) pair.
   - Return top 20 by reranker score.

4. **Optional ColBERT late-interaction reranking** (toggle in UI; default off):
   - Compute multi-vector embeddings on-the-fly for the top 20 (since we have them pre-indexed in `jobs_multivec`, retrieve them).
   - Compute MaxSim between query token vectors and each chunk's token vectors.
   - Rerank top 20 by MaxSim score.

5. **Parent-chunk hydration** → top 5–10 parent chunks:
   - For each retrieved child chunk, fetch its parent.
   - Deduplicate (child chunks from the same parent collapse).
   - Return top 5–10 unique parents.

6. **Optional context summarization** (toggle; auto-on if context > 4k tokens):
   - For each parent chunk, Qwen2.5-7B produces a 2–3 sentence summary preserving the salary, seniority, key skills, and overall fit signal.

7. **Generation**:
   - Final LLM call passes original query + summarized contexts + relevant predictions (predicted salary band, seniority match) to Qwen2.5-7B.
   - Output: ranked job recommendations with rationale.

### NL→SQL endpoint

Separate from the retrieval path. Used for analytics queries like "what's the median salary for senior MLE roles in NYC over the last 90 days?"

- Qwen2.5-7B receives the query + a fixed schema description for the curated parquet (table + column list with types and descriptions).
- It outputs a DuckDB SQL query.
- **Safety layer (mandatory, never skip):**
  - Parse the SQL with `sqlglot`.
  - Reject if it references tables outside an allowlist.
  - Reject if it references columns outside an allowlist.
  - Reject if it contains DDL or non-SELECT statements.
  - Cap result rows (LIMIT 1000) and execution time (5s).
- Execute against DuckDB.
- Return results + the executed SQL + a 2-sentence explanation generated by the LLM.

### Eval harness

Located at `eval/`. Run via `python -m eval.run_retrieval_eval`.

- `eval/queries.jsonl` — 50–100 hand-labeled queries, seeded from the builder's own resume and target queries. Each query has a list of relevant `job_id`s.
- Metrics computed: recall@5, recall@10, MRR, nDCG@10.
- Run separately for each retrieval variant: dense-only, sparse-only, hybrid, hybrid+rerank, hybrid+rerank+ColBERT, hybrid+rerank+ColBERT+HyDE.
- Output: a CSV in `reports/eval/` and a markdown table in the README, updated whenever the eval set or models change.

---

## 9. Repo structure

```
na-tech-jobs/
├── README.md                    # public-facing, links to Space, narrative, eval table
├── CLAUDE.md                    # this file
├── pyproject.toml               # uv-managed
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── ci.yml               # lint + test on PR
│       ├── ingest.yml           # weekly cron (Sundays 02:00 UTC)
│       ├── drift.yml            # weekly cron (Mondays 03:00 UTC)
│       └── retrain.yml          # monthly cron (1st @ 04:00 UTC)
├── ingestion/
│   ├── __init__.py
│   ├── companies.yaml           # registry of companies + ATS handles
│   ├── extractors/
│   │   ├── base.py              # ABC + retry/backoff/normalize logic
│   │   ├── greenhouse.py
│   │   ├── lever.py
│   │   ├── ashby.py
│   │   ├── workable.py
│   │   ├── smartrecruiters.py
│   │   └── workday.py           # tenant-config-driven
│   ├── normalize.py             # title/location/currency/period
│   ├── dedup.py
│   ├── quality.py               # Pandera schemas + validation
│   ├── orchestrator.py          # runs extractors, writes snapshot, calls quality
│   └── push_to_hub.py
├── curated/
│   ├── build.py                 # dedup curated table from snapshots
│   ├── enrich.py                # adds model predictions
│   └── duckdb_views.sql         # named views for common queries
├── models/
│   ├── salary/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── README_template.md
│   ├── seniority/
│   ├── role_family/
│   ├── skills/
│   └── embeddings/
├── rag/
│   ├── chunking.py
│   ├── embedder.py              # bge-m3 wrapper, ZeroGPU-friendly
│   ├── reranker.py
│   ├── colbert.py               # late-interaction MaxSim
│   ├── hyde.py
│   ├── summarizer.py
│   ├── nl2sql.py                # includes SQL safety layer
│   ├── pipeline.py              # full retrieval orchestration
│   └── qdrant_client.py         # local-mode setup, schema, upsert helpers
├── app/
│   ├── main.py                  # Gradio entrypoint
│   ├── tabs/
│   │   ├── matcher.py           # resume → top-k jobs
│   │   ├── search.py            # NL query → ranked jobs
│   │   ├── analytics.py         # NL→SQL UI
│   │   └── dashboard.py         # drift + market trends
│   ├── resume_parser.py
│   └── styles.css
├── eval/
│   ├── queries.jsonl            # hand-labeled
│   ├── run_retrieval_eval.py
│   ├── metrics.py
│   └── reports/                 # output dir
├── monitoring/
│   ├── drift.py                 # Evidently runner
│   ├── pipeline_health.py       # extractor success/failure log
│   └── alerts.py                # Discord webhook
├── infra/
│   ├── persistent_storage.md    # what lives in /data on Spaces
│   └── secrets.md               # which env vars are needed
├── tests/
│   ├── ingestion/
│   ├── models/
│   ├── rag/
│   └── eval/
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_baseline_salary.ipynb
│   └── …                        # exploratory only, not load-bearing
└── scripts/
    ├── bootstrap_companies.py   # build initial companies.yaml from seed list
    ├── label_seniority.py       # CLI for hand-labeling
    └── label_eval_queries.py    # CLI for building the eval set
```

---

## 10. Phased build plan

The phases are sized so that every phase ends with **something demonstrably working** that could be shown to a recruiter (even if not yet polished). The total target is ~10 weeks at full focus, ~14–16 weeks at moderate pace. Don't skip phase exit criteria — they're the gate.

### Phase 0 — Project setup (Week 1, ~3 days)

**Goal:** repo scaffolded, CI green, can deploy a "hello world" Gradio Space.

Tasks:
- Initialize repo with `uv init`, set up `pyproject.toml` with the locked dependency list.
- Configure pre-commit (ruff format + lint).
- Set up `.github/workflows/ci.yml` running `pytest` and `ruff` on every PR.
- Create empty HF Dataset repo, HF Model org/namespace, HF Space (Gradio template).
- Wire the Space to the GitHub repo via Spaces' git integration.
- Set Discord webhook env var for alerting; verify end-to-end with a test message.
- Configure persistent storage on the Space (request 20GB; mount at `/data`).
- Document all secrets in `infra/secrets.md`: `HF_TOKEN`, `DISCORD_WEBHOOK_URL`, anything else.

Exit criteria:
- `git push` triggers CI, CI passes.
- The Space is live (even if it's a hello-world Gradio).
- A Discord notification fires on a test event.

### Phase 1 — Ingestion v1 (Weeks 1–2, ~7 days)

**Goal:** weekly ingestion of 3 ATS providers (Greenhouse + Lever + Ashby) for ~50 seed companies, producing versioned parquet snapshots in the HF Dataset repo.

Tasks:
- Build `ingestion/companies.yaml` with 50 seed companies (mix of US + Canadian, mix of providers). Source candidates: YC NA companies, Canadian tech scale-ups, Toronto/Vancouver/SF/NY clusters.
- Implement `ingestion/extractors/base.py` with retry, rate limiting, schema normalization.
- Implement `greenhouse.py` (`https://boards-api.greenhouse.io/v1/boards/{handle}/jobs?content=true`).
- Implement `lever.py` (`https://api.lever.co/v0/postings/{handle}?mode=json`).
- Implement `ashby.py` (`https://api.ashbyhq.com/posting-api/job-board/{handle}`).
- Implement `normalize.py` for title, location, currency, period.
- Implement `dedup.py` (against the prior week's snapshot, by `id`).
- Implement `quality.py` with a Pandera schema.
- Implement `orchestrator.py` — runs all extractors in parallel (asyncio + httpx), aggregates, dedups, validates, writes snapshot.
- Implement `push_to_hub.py` — git commit to the Dataset repo with a meaningful message.
- Wire it all into `.github/workflows/ingest.yml` running every Sunday 02:00 UTC.

Exit criteria:
- `python -m ingestion.orchestrator` runs end-to-end locally and writes a parquet.
- The GitHub Action runs on schedule and produces a commit to the Dataset repo.
- Snapshot has 5–10k rows, schema validates, dedup is correct.
- Discord notification fires on success and failure.

### Phase 2 — Curated layer + first model (Week 3, ~7 days)

**Goal:** a clean curated table, the salary regressor end-to-end, MLflow tracking, model on HF Hub.

Tasks:
- Implement `curated/build.py` — DuckDB query that produces the curated table from snapshots.
- Implement `curated/duckdb_views.sql` with views: active jobs, by source, by country, etc.
- Implement `models/salary/train.py` — feature engineering, XGBoost training with Optuna, MLflow logging.
- Implement `models/salary/predict.py` — inference wrapper.
- Push trained model to a HF Model repo (`<owner>/na-tech-jobs-salary-v1`).
- Generate a model card from MLflow metadata.

Exit criteria:
- Curated table builds reproducibly from the latest snapshot.
- Salary model trains, achieves reasonable MAE on US-disclosed test set (target: MAE < $25k USD/year).
- Model + card are live on HF.
- Stratified eval (by country, by source) is in the model card.

### Phase 3 — First end-to-end deploy (Week 4, ~5 days)

**Goal:** Gradio app live on Spaces with a working salary prediction endpoint. **The first version that's resume-worthy on its own.**

Tasks:
- Implement `app/main.py` Gradio shell with one tab.
- Implement `app/tabs/search.py` v0 — a basic keyword search over the curated parquet (no embeddings yet).
- Add a `/predict/salary` endpoint that loads the model from HF Hub on startup and serves predictions for an arbitrary job description input.
- Configure ZeroGPU (`@spaces.GPU(duration=60)` on inference functions).
- Enable always-on, verify cold-start UX.

Exit criteria:
- Space is live, always-on, with a working keyword search and salary prediction.
- A recruiter clicking the demo link sees something that works in <5 seconds.
- README v1 written (project description, narrative, demo link, eval table placeholder).

### Phase 4 — Multi-model + payload enrichment (Weeks 5–6, ~10 days)

**Goal:** seniority + role-family classifiers trained, all models score the curated table, payloads ready for retrieval.

Tasks:
- Build `scripts/label_seniority.py` CLI for hand-labeling 500 examples. Spend ~half a day on this.
- Build the same for role family.
- Implement `models/seniority/train.py` — DeBERTa-v3-base + LoRA on weakly supervised training labels.
- Implement `models/role_family/train.py`.
- Implement `models/skills/predict.py` — NuExtract zero-shot extraction with skill taxonomy normalization.
- Implement `curated/enrich.py` — runs all four models over the curated table, writes back with versioned columns.
- Push all models to HF Hub with model cards.
- Add Workable + SmartRecruiters extractors (now's the time to expand provider coverage).

Exit criteria:
- All four models live on HF with model cards.
- Enriched curated table has versioned prediction columns.
- Provider coverage expanded to 5 ATS providers + 80–100 companies.
- Hand-labeled test sets exist in `eval/` and report metrics in model cards.

### Phase 5 — Retrieval stack (Weeks 7–8, ~10 days)

**Goal:** hybrid dense + sparse retrieval with cross-encoder reranking, indexed in Qdrant, powering the matcher UI.

Tasks:
- Implement `rag/chunking.py` with parent-child recursive splitting.
- Implement `rag/embedder.py` — bge-m3 wrapper with ZeroGPU decoration.
- Implement `rag/qdrant_client.py` — local-mode setup, two collections, schema, upsert helpers.
- Build a one-shot indexing script: chunk all curated jobs, embed (dense + sparse), upsert into Qdrant.
- Implement `rag/reranker.py` with bge-reranker-v2-m3.
- Implement `rag/pipeline.py` — orchestrates first-pass hybrid retrieval (RRF fusion) + cross-encoder rerank.
- Implement `app/resume_parser.py` — pypdf + LLM-based skill extraction.
- Implement `app/tabs/matcher.py` — resume → top-k jobs.
- Add filtering by predicted_salary, seniority_label, country to the UI.

Exit criteria:
- 240k+ chunks indexed in Qdrant on persistent storage.
- Matcher UI works: paste resume, get ranked job list with predictions visible per job.
- Latency under 5 seconds for default path.
- Index re-build is automated (script run after enrichment).

### Phase 6 — ColBERT, HyDE, eval harness (Weeks 9–10, ~10 days)

**Goal:** the multi-variant retrieval comparison, with a hand-labeled eval set producing real numbers.

Tasks:
- Build `eval/queries.jsonl` — hand-label 50–100 queries seeded from the builder's resume and target searches. Spend a full day on this.
- Implement `eval/metrics.py` (recall@k, MRR, nDCG).
- Implement `eval/run_retrieval_eval.py` — runs each retrieval variant against the eval set, outputs CSV + markdown.
- Index multi-vector embeddings into `jobs_multivec` collection (compute upfront, store compressed).
- Implement `rag/colbert.py` — fetch multi-vec from Qdrant, compute MaxSim.
- Implement `rag/hyde.py` — Qwen-driven hypothetical doc generation.
- Add toggles in UI for HyDE on/off and ColBERT on/off.
- Run the full eval, populate the README's eval table.

Exit criteria:
- Eval table in README shows recall@10, MRR, nDCG for: dense-only, sparse-only, hybrid, hybrid+rerank, hybrid+rerank+ColBERT, hybrid+rerank+ColBERT+HyDE.
- Statistical commentary in README on what each addition gains and at what latency cost.

### Phase 7 — LLM layer + NL→SQL (Week 11, ~7 days)

**Goal:** Qwen2.5-7B serving HyDE, summarization, NL→SQL, and the final generative answer in retrieval.

Tasks:
- Implement `rag/summarizer.py` — Qwen prompts for parent-chunk summarization.
- Wire summarization into `rag/pipeline.py` — auto-on when context > 4k tokens.
- Implement `rag/nl2sql.py` with:
  - Schema-aware prompting.
  - sqlglot-based parsing.
  - Allowlist enforcement (tables, columns, statement type).
  - Row + time caps.
- Implement `app/tabs/analytics.py` — NL → SQL → DuckDB → results table.
- Add the final generation step to the matcher (LLM produces a paragraph rationale per top job).

Exit criteria:
- Analytics tab works on natural-language queries: "median senior MLE salary in Toronto last 90 days," etc.
- Rejected queries (out of scope, unsafe) return a clear error rather than failing silently.
- Matcher returns results with LLM-generated per-job rationale.

### Phase 8 — Drift dashboard, observability, retraining automation (Week 12, ~7 days)

**Goal:** the project is **operationally alive** — drift detected weekly, retraining automated monthly, pipeline health visible.

Tasks:
- Implement `monitoring/drift.py` — Evidently AI configured for the curated table's key features and the prediction distributions.
- Wire `.github/workflows/drift.yml` — Mondays 03:00 UTC, writes report to `reports/drift/<date>.html`.
- Implement `monitoring/pipeline_health.py` — log per-extractor success/failure stats during ingest; produces a summary JSON.
- Implement `app/tabs/dashboard.py` — drift charts, market trend charts (salary distributions over time, top tech stacks rising, role family proportions), pipeline health card.
- Implement `.github/workflows/retrain.yml` — full retraining pipeline, champion/challenger gate, auto-promotion if winning.
- Configure drift threshold breach to set a `priority=true` flag on next retrain.

Exit criteria:
- Drift dashboard live, updating weekly.
- Monthly retraining runs end-to-end without intervention.
- Pipeline health card on dashboard shows last successful run + per-extractor status.

### Phase 9 — Polish, docs, launch (Weeks 13–14, ~7 days)

**Goal:** the project is **interview-ready**.

Tasks:
- Rewrite README v2: lead with the narrative, screenshots, demo GIF, eval table, architecture diagram, links to model cards, links to dataset, methodology notes.
- Generate an architecture diagram (Excalidraw or Mermaid) committed to the repo.
- Record a 3–5 minute Loom walkthrough — pin to top of README.
- Write a Medium / personal blog post: "Building a production ML platform for the senior tech-hiring market on a $9 budget." Include eval numbers, design tradeoffs, latency budgets.
- Polish all model cards.
- Write the public dataset card (HF dataset README).
- Run a final QA pass: every endpoint, every tab, every link.
- Publish on LinkedIn with a clear narrative tying the project to the senior DS skills it demonstrates.

Exit criteria:
- A recruiter who clicks the LinkedIn post → README → demo link → blog post sees a coherent senior-level narrative inside 5 minutes of attention.
- Every component is documented enough that someone else could reproduce it.

### Stretch goals (post-v1, do these in Phase 10+ if energy permits)

- Quebec French postings.
- EU comparison set.
- Email digests / RSS feed of new high-fit jobs for the user's resume.
- Browser extension that overlays predicted salary on any job posting.
- Fine-tune a small model on the curated dataset for a named-entity-aware embedder optimized for tech jobs.
- API tier with rate-limited public access (would need user auth).

---

## 11. Risks and mitigations

These are known risks. Address each before / during the relevant phase. Don't pretend they don't exist.

### High-impact

**Workday extractor brittleness.** Each Workday tenant has a unique URL and schema variations. Mitigation: maintain a per-tenant config in `companies.yaml` (tenant URL, search path overrides, field mappings); accept that some tenants will break and a manual fix is needed; start with a tested set of 30–50 stable tenants and grow slowly.

**Salary data sparsity / bias.** US disclosure laws give us 50–70% disclosure on US postings, less on Canadian. The salary regressor's training distribution is skewed. Mitigation: stratified eval (report MAE per country and per source), explicit framing in model card and README ("predicts salary the way disclosing companies in our corpus would price this role"), don't claim the model predicts ground-truth market rate for non-disclosing roles.

**Label noise on seniority and role family.** Rule-based extraction from titles is noisy. Mitigation: hand-labeled clean test sets (500 each) — the cleanness of the test set is more important than the cleanness of the training set. Report metrics on the hand-labeled set, not the noisy set.

**ZeroGPU latency on the full RAG path.** HyDE + embed + retrieve + rerank + ColBERT + summarize + generate could be 15–30s per query. Mitigation: ColBERT and HyDE are off by default; UI labels them as "quality mode: slow"; default path stays under 5s. Run a latency budget after Phase 6 to verify.

**Re-indexing on retraining is operational pain.** Every model retraining changes the predictions stored in Qdrant payloads. Mitigation: monthly cadence (not weekly), versioned prediction columns kept in payloads (don't delete old versions immediately), background re-indexing job that can run for hours without affecting serving.

### Medium-impact

**Drift signal weak on small weekly deltas.** Mitigation: 4-week rolling baseline, monthly trend analysis on top of weekly checks. Frame the dashboard accordingly.

**Eval harness construction is real work.** Mitigation: budget a full day for building the labeled query set in Phase 6. Seed from the builder's own resume and target queries.

**NL→SQL hallucinations.** Mitigation: sqlglot-based safety layer is mandatory. Allowlist tables and columns. Reject DDL. Cap rows and time. Always show the executed SQL to the user so they can verify.

**Cold start on Spaces with multiple models.** Mitigation: always-on enabled. Models preloaded to CPU at Space startup, moved to GPU per-request.

**Persistent storage cost growth.** Mitigation: store only Qdrant index + drift report HTML on persistent disk; let model weights re-pull from HF Hub on cold start (HF caches them locally after first download).

### Low-impact but worth tracking

**Pipeline silent failure.** Mitigation: Discord webhook on every Action run, dashboard card showing last successful ingest timestamp.

**Demo abuse / quota burn.** Mitigation: monitor ZeroGPU usage; if quotas become a problem, add simple per-IP rate limiting or queue.

**License compliance.** All chosen models are Apache-2.0 or MIT. Verify before any new model addition.

---

## 12. Out of scope for v1

These are explicit non-goals. Do not start work on them in v1, even if they seem like easy wins.

- **LinkedIn / Indeed / Glassdoor scraping** — ToS-prohibited.
- **Quebec French postings** — English-only MVP.
- **EU coverage** — North America only for v1.
- **Mobile app or native mobile UI** — Gradio only.
- **Multi-tenant user accounts** — single-user implicit. Optional auth comes later if API tier is built.
- **Real-time streaming ingest** — weekly batch only.
- **Premium / paid features** — fully open and free.
- **Custom LLM fine-tuning** — using foundation models off-the-shelf is sufficient for v1.

---

## 13. Conventions for Claude Code

When working on this project, Claude Code should:

### Code style
- Python 3.11+ exclusively. Use modern syntax (`match` statements, `|` for union types, etc.).
- Type hints on all function signatures and class attributes.
- Docstrings on all public functions and classes (Google style).
- `ruff` is the formatter and linter. Run `ruff format && ruff check --fix` before committing.
- Prefer composition over inheritance. Prefer dataclasses or pydantic models over dicts.

### Testing
- Every new module gets a corresponding test file in `tests/`.
- Use pytest. Use parametrize for table-driven tests.
- Aim for tests that verify behavior, not implementation. Don't mock more than needed.
- Run `pytest tests/` before pushing.

### Git
- Conventional commits format: `feat(ingestion): add Greenhouse extractor`, `fix(rag): correct chunk overlap`, `docs(readme): add eval table`.
- Branch per feature: `feat/ingestion-greenhouse`, `fix/qdrant-payload-types`.
- PRs include: what, why, screenshots if UI, test results.

### Working with this project
- Always read `CLAUDE.md` (this file) before starting a new work session.
- The phased plan in section 10 is the source of truth for what to build next. Don't skip ahead unless the user explicitly asks.
- When making architectural decisions not covered here, write the decision into this file alongside the change. Treat this file as a living document.
- When adding a new dependency, justify it in the PR (why this lib, why not the existing options, license check).
- Ask before introducing any major new framework (LangChain agents, Ray, Dagster, etc.). The locked stack is deliberate.
- For any LLM-touching code, add a unit test that mocks the LLM response — never let CI depend on a live LLM call.
- For any model that goes to HF Hub, write the model card alongside the training code.

### When in doubt
- Default to the simpler option.
- Default to the option that better demonstrates the senior DS signal (see section 2).
- Default to writing fewer lines of code.
- Default to making the project work end-to-end before optimizing any one component.
- If unsure about scope, ask the user before building.

---

## 14. Success criteria

### Per-phase exit criteria
Listed inline in each phase in section 10.

### Project-level success
- The Space is live, always-on, and a recruiter clicking the demo link sees a working product within 5 seconds.
- The eval table in the README has real numbers from the labeled query set, comparing four+ retrieval variants.
- All four production models are on HF Hub with non-trivial model cards.
- The dataset is on HF Hub with weekly updates and a status badge showing last successful ingest.
- The drift dashboard updates weekly.
- A blog post and Loom walkthrough are linked from the README.
- The builder has actually used the tool to find at least one real job lead.

### Final sanity check before declaring v1 shipped
- Open the README in incognito mode. Read it as a recruiter who has 90 seconds. Does it tell the senior DS story clearly? If not, fix the README.
- Click the demo link. Within 30 seconds, can you do something useful (search, get a salary prediction, ask a NL question)? If not, fix the UX.
- Look at the model cards. Could a senior interviewer pull real critique signal from them? If not, deepen them.

---

## 15. The elevator pitch

For interviews, LinkedIn, and the README. Memorize this; it's the project's reason for existing.

> "I built a production ML platform for the North American senior tech-hiring market. It runs weekly ATS ingestion across Greenhouse, Lever, Ashby, Workable, SmartRecruiters, and a curated Workday tenant list, producing a versioned dataset on the Hugging Face Hub. On top of that, I trained a salary regressor (XGBoost on tabular features plus bge-m3 embeddings), seniority and role-family classifiers (DeBERTa-v3 with LoRA), and a skill extractor (NuExtract). All four models enrich a Qdrant payload that powers a hybrid + late-interaction RAG layer — bge-m3 for dense, sparse, and ColBERT-style multi-vector retrieval, with cross-encoder reranking, optional HyDE, and Qwen2.5-7B for context summarization and natural-language analytics. Drift detection runs weekly, retraining runs monthly with champion/challenger promotion. The whole thing deploys to a $9 Hugging Face Pro Space with always-on enabled, using ZeroGPU for inference. I built it because I needed it for my own senior DS job search."

---

*This file is the single source of truth for the project. Update it when reality changes.*
