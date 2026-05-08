# Maintenance log

Running list of known issues, debt, and follow-ups not large enough to be their
own phase. New entries go at the **top** of each section. Each entry says:

- **what's wrong** (or what's missing),
- **why it's not blocking** (or why we accepted it),
- **target phase** for the fix,
- **status**: `open`, `in-progress`, `resolved`, `wontfix`.

When an item is resolved, leave it in place with the resolution date — the log
doubles as a project-history artifact.

Conventions:
- This file is hand-maintained. Don't generate from CI / git.
- Keep entries terse. If something needs more than a paragraph, write it up in
  CLAUDE.md or a phase exit note instead.
- Cross-reference CLAUDE.md sections when relevant: `(CLAUDE.md §11)`.

---

## Open

### Data quality

- **Lever `interval=OneTime` mapped to `year`.** Only `per-year-salary` and
  the `*-salary` variants are documented. `OneTime` is a guess based on a
  handful of postings — could equally mean a one-off bonus rather than annual.
  - Target: **opportunistic** — when we actually see one in the wild, inspect
    and map correctly. Until then, the wrong mapping affects 0 rows.
  - Status: `open`.

- **`offers_visa_sponsorship` at 0.3% fill, `direct_reports_count` near zero.**
  These features are sparse in regex coverage because most descriptions don't
  explicitly state sponsorship policy or report counts. Phase 1b NuExtract
  should bump both significantly (LLM can read the contextual cues regex misses).
  - Target: **Phase 1b** (NuExtract Tier 2 wiring).
  - Status: `open`.

- **Greenhouse double-encoded HTML in `description_md` was unprocessed.**
  Greenhouse's API returns content as `&lt;h2&gt;`-style escaped HTML, which
  caused `markdownify` to emit literal HTML rather than markdown. Fixed in
  Step 1a by `html.unescape` before `markdownify`. The 2026-05-08 backfill
  re-processed the entire snapshot. **Resolved 2026-05-08.**

### Companies registry

- **High-volume tech employers still missing.** After the 1.5 expansion the
  remaining gaps are mostly Workday-only or proprietary careers pages:
  - DoorDash, Snowflake, OpenAI, Klarna, Etsy, Wayfair, Coinbase, Shopify,
    Lightspeed, Hugging Face, Verily, Tempus, 23andMe, Wiz, Snyk, Aurora,
    Skydio, Hims, Hinge Health, Confluent, Zendesk, Box, Hopper, Top Hat,
    Ada, ApplyBoard, Bench Accounting, Clio.
  - Target: **Phase 4** — Workday extractor unlocks most of these (Snowflake,
    Etsy, Wayfair, Shopify, DoorDash, Coinbase use Workday). Hugging Face +
    OpenAI run their own careers pages; revisit if/when they expose JSON.
  - Status: `wontfix-until-phase-4`.

- **Why not non-tech sectors (banks, insurance, retail, healthcare systems)?**
  Out of scope for v1 by design (CLAUDE.md §1 + §3 lock #2). Most concentrate
  in Workday, which lands in **Phase 4**. Adding them now creates two problems:
  (a) dilutes the "senior tech-hiring" narrative the project is built around,
  (b) Workday extractor isn't built yet. Wait for Phase 4 then add 30-50
  Workday tenants per CLAUDE.md §3 lock #2. Recorded here so we don't keep
  re-litigating.
  - Status: `wontfix-until-phase-4`.

### Schema / normalization

- **Naive seniority + role-family classifiers.** Title-regex based; CLAUDE.md
  §7 says these get replaced by DeBERTa-v3 + LoRA in Phase 4. Current Phase 1
  output has `role_family_extracted=Other` for 4,538 of 6,709 rows (68%) —
  expected (the regex only labels obvious matches), but means downstream
  consumers should treat the extracted column as weak supervision, not
  ground truth.
  - Target: **Phase 4** (per CLAUDE.md §10).
  - Status: `open`.

- **Hardcoded FX rate (1 CAD = 0.73 USD).** Frozen in `ingestion/normalize.py`.
  Acceptable for v1 — salary modelling works on USD-yearly normalized values
  and a stale rate biases CA salaries by ~3-5%, well below the model's MAE
  target of <$25k. CLAUDE.md §11 doesn't flag it but it's worth noting.
  - Target: **opportunistic** — pull a daily FX from a free source (e.g.
    Bank of Canada noon rate) on each ingest run. Cache in the snapshot
    metadata so older parquets remain reproducible.
  - Status: `open`.

### CI / ops

- **No drift detection yet.** CLAUDE.md §10 schedules this for Phase 8. The
  weekly ingest currently writes a `reports/ingestion/<date>.json` but no
  Evidently report. Without drift checks we won't notice if (e.g.) a major
  ATS schema change starts breaking salary parsing.
  - Target: **Phase 8** (per CLAUDE.md §10).
  - Status: `open`.

- **Workflow Node.js 20 deprecation warnings.** GitHub Actions deprecates
  Node 20 actions on 2026-09-16. `actions/checkout@v4` and `astral-sh/setup-uv@v3`
  are flagged. Non-blocking until then.
  - Target: **opportunistic** — bump to v5 / v4 of those actions when stable
    versions ship. Or add `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` per workflow.
  - Status: `open`.

- **Cache restore intermittently 400s on the deploy and CI workflows.** Doesn't
  affect job success; the cache just rebuilds. GitHub Actions caching service
  blip during the Phase 0 / Phase 1 push window — not actionable on our side
  but worth noting if it persists.
  - Status: `open` (monitor only).

---

## Resolved

### 2026-05-08 — Phase 2 Step 3: salary regressor (six-tier ladder)

- **Six tiers from constant baseline to XGBoost+Optuna**, all evaluated on the
  frozen 80/20 train/test split (test n=1,226). Per CLAUDE.md §10 + LITERATURE_REVIEW.md §16.
- **5-fold CV-MAE on the training set** runs alongside test-MAE for every tier — the
  substitute for Cohen-style power analysis (LITERATURE_REVIEW.md §15.3 #15).
- **Final leaderboard**:

  | Tier | test-MAE | test 95% CI | CV-MAE | CV 95% CI | MAPE | R² log |
  |---|---|---|---|---|---|---|
  | 0 constant | $60,509 | $57k–$64k | $62,279 | $61k–$64k | 33.7% | 0.000 |
  | 1 stratified | $59,589 | $56k–$63k | $61,623 | $60k–$63k | 32.9% | 0.045 |
  | 2 Mincer OLS | $51,322 | $48k–$54k | $52,041 | $50k–$53k | 27.4% | 0.283 |
  | 3 Ridge | $43,199 | $41k–$45k | $42,179 | $41k–$43k | 23.3% | 0.462 |
  | 4 Random Forest | $35,935 | $34k–$38k | $37,016 | $36k–$38k | 19.0% | 0.615 |
  | **5 XGBoost+Optuna** | **$29,091** | **$27k–$31k** | **$30,533** | **$29k–$32k** | **14.7%** | **0.730** |

- **Test-MAE and CV-MAE agree within 5% across every tier**: no evidence of
  overfitting or lucky-test-draw.
- CLAUDE.md §10 target ($25k MAE) **not yet hit**: the closing gap is for
  bge-m3 description embedding in Phase 5.
- Winning model + 5 artifacts pushed to https://huggingface.co/arjun10g/na-tech-jobs-salary-v1
  (commit `dae4f6c`).
- New module tree: `models/salary/{dataset,encode,baselines,linear,forest,xgb,
  eval,train,predict,model_card}.py` + `scripts/publish_salary_model.py`. 30
  unit tests added across encoders / baselines / eval (now 183 passing).
- Ruff per-file ignores added for `models/**` and `tests/models/**` to allow
  the sklearn convention of `X` for feature matrices and `y` for targets.

### 2026-05-08 — Phase 2 Steps 2 + 2.5: curated layer + statistical audit + literature review

- **Curated layer** (`curated/build.py`, `curated/duckdb_views.sql`):
  DuckDB stack across snapshots, computes first/last_seen_at + times_seen,
  emits `curated/jobs.parquet` (active = in latest snapshot) and
  `curated/jobs_history.parquet` (every job ever seen). 5 unit tests cover
  single-snapshot, continuing-job, delisted-job, brand-new-job dedup. Pushed
  to `arjun10g/na-tech-jobs` under `curated/`.
- **Statistical EDA audit** (`eda/audit.py` + `eda/report.py`): single-command
  `uv run python -m eda.audit` produces `data/eda/{report.md, metrics.json,
  plots/*.png}`. Sections: schema + dtype classification (49 cols / 9 roles),
  univariate distributions, missingness with chi-square MAR diagnostics,
  target deep-dive (raw + log10 + stratified), bivariate (Pearson/Spearman +
  ANOVA F + Welch t), multicollinearity (VIF + heatmap + condition number),
  outlier audit, MNAR / omitted-variable / transformation discussion, modelling
  implications. First audit snapshot committed at `eda/reports/2026-05-08/`.
- **Literature review** ([LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)): 17
  sections / ~600 lines. Per-predictor treatment with citations (Tukey, HTF,
  Kuhn & Johnson, Heckman, Pargent et al, Cerda & Varoquaux, Chen & Guestrin,
  Mitchell et al model cards). §14 is the recommendations table for the Step 3
  regressor. §15 added an ideal EDA pipeline + self-audit: 15 of 20 stages
  done, 1 missing (PCA/multivariate), 4 partial.

### 2026-05-08 — Phase 2 Step 1b: NuExtract Tier 2 wired, dormant by default

- **Real `NuExtract` wrapper landed** at `ingestion/feature_extraction/llm/nuextract.py`.
  Lazy-loads `numind/NuExtract-tiny-v1.5`, picks MPS / CUDA / CPU by availability,
  uses left-padded batched generation. `run_batch()` is the throughput path used
  by `cascade.extract_features_batch()`. 30 NuExtract-specific tests with the
  model fully mocked, including a regression that aligned the batch-output index
  back to the input index.
- **Backfill path supports `--use-llm`, `--sample-llm N`, `--llm-batch-size`** via
  `scripts/backfill_features.py`. A two-pass mode (regex everywhere, then LLM
  on a random subset) keeps partial-coverage runs simple.
- **Throughput reality check (M-series Mac, MPS, float16)**:
  - batch=1  → 8.0 s/row
  - batch=8  → 4.6 s/row (after `MAX_INPUT_CHARS=2000`)
  - batch=16 → 3.6 s/row → 12.3 hrs for the 12k snapshot
- **Decision: keep Tier 2 dormant by default.** `LLM_ELIGIBLE_FIELDS = frozenset()`
  in `cascade.py`. The fields it would fill (industry_experience, team_or_department,
  prose-only tech stack mentions, more nuanced sponsorship / citizenship language)
  don't feed the salary regressor (Step 3) and the bge-m3 description embedding
  (Phase 5) carries the same semantic signal where it matters. Cost-benefit
  doesn't justify the 12+ hr backfill yet.
- **Re-enable path documented**: populate `LLM_ELIGIBLE_FIELDS` and run
  `uv run python -m scripts.backfill_features --use-llm --llm-batch-size 16`,
  ideally on a GPU box (HF Jobs A10G ≈ 30 min ≈ $0.50, or ZeroGPU on the
  existing Pro Space). Local Mac MPS is feasible overnight.

### 2026-05-08 — Phase 2 Step 1a: feature-extraction cascade

- **Built the regex-first cascade** at `ingestion/feature_extraction/`. Tier 1
  ships seven regex modules (salary, experience+education, requirements,
  remote+schedule, comp-extras, contract+quality+language+manager,
  tech_stack). Tier 2 (NuExtract) is stubbed; lands in Step 1b. Per-field
  provenance flows through `extraction_meta`.
- **Schema expanded** with 22 new feature columns + `extraction_meta` +
  `extraction_version` on `CanonicalJob`. Pandera validates them; nullable
  Int64 / pandas BooleanDtype / object lists roundtrip cleanly through parquet.
- **Salary mining live**: 0% → **49.8% disclosure rate** on the 12,334-row
  snapshot, in the CLAUDE.md §11 expected band of 50-70%.
- **Other key fill rates** (post-backfill): min_years_experience 81.8%,
  offers_equity 64.9%, tech_stack 64.7%, remote_policy 44.7% (was 19.6%
  pre-cascade), bonus_mentioned 36.3%, contract_type 34.4%, manager_role 25.5%,
  min_education 25.2%, requires_security_clearance 9.8% (concentrated at
  Anduril/SpaceX as expected).
- **HTML→MD bug fixed** — Greenhouse double-encodes content; needed
  `html.unescape` before `markdownify`.
- **123 unit tests** covering positive + negative cases for every Tier 1
  extractor, including critical regressions (the "up to 100% match on 401k"
  false-positive on travel percent).
- Snapshot pushed: `arjun10g/na-tech-jobs` commit `c2d9db0`.

### 2026-05-08 — Phase 1.5 expansion

- **Pruned dead seed handles + added 30 new tech-adjacent companies.**
  Built `scripts/probe_handles.py` to probe (provider, handle) candidates in
  parallel, kept the 65 that returned ≥1 job. Re-ingest produced **12,334 rows**
  (up from 6,709, +84%) across 65 boards (47 Greenhouse, 16 Ashby, 2 Lever)
  with 0 company failures. Notable new entries: Anduril (1,888), SpaceX (1,684),
  Databricks (818), Jane Street (215), Plaid + Ramp via Ashby, Spotify via
  Lever, Notion + Whatnot + Mistral via Ashby. Snapshot pushed to
  `arjun10g/na-tech-jobs` commit `220ad34`. Closed both "dead seed handles"
  and "tech-adjacent companies missing" items.
