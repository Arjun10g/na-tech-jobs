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

- **0% structured salary disclosure on Phase 1 snapshot.** All 6,709 rows came
  back with `salary_disclosed=false` because Greenhouse/Lever/Ashby rarely
  populate the structured pay-range fields — most boards bury salary in the
  description text (e.g. "$135,000 - $180,000 USD"). Schema handles sparse
  salaries (nullable + `salary_disclosed` bool), so this doesn't break Phase 2,
  but the regressor will train on a thin disclosed set unless we add inline
  text mining. CLAUDE.md §11 calls this out as a known risk.
  - Target: **Phase 2** — add a regex pass in `ingestion/normalize.py` (or a
    new `ingestion/salary_mining.py`) that runs when structured fields are empty.
    Regex variants to support: `$XXX,XXX - $YYY,YYY`, `$XXXk - $YYYk`,
    `USD XXX,XXX to YYY,YYY`, `CAD …`. Keep `salary_disclosed=true` only when
    parse succeeds with high confidence.
  - Status: `open`.

- **Lever `interval=OneTime` mapped to `year`.** Only `per-year-salary` and
  the `*-salary` variants are documented. `OneTime` is a guess based on a
  handful of postings — could equally mean a one-off bonus rather than annual.
  - Target: **opportunistic** — when we actually see one in the wild, inspect
    and map correctly. Until then, the wrong mapping affects 0 rows.
  - Status: `open`.

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
