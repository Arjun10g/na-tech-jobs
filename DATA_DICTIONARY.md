# Data Dictionary — `data/curated/jobs.parquet`

> Reference for every column in the curated layer. Each row says what the
> field _is_, where it comes from, how full it is, and whether it's an
> input to the salary regressor. **Companion docs**:
> [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) for _how_ to treat each
> predictor; [`eda/reports/2026-05-08/report.md`](eda/reports/2026-05-08/report.md)
> for the audit numbers; [`ingestion/schema.py`](ingestion/schema.py) for
> the source-of-truth Pydantic model.

**Snapshot stats** (curated layer, 2026-05-08): 12,334 rows × 49 columns;
US 95% / CA 5%; ~46% have a disclosed salary.

**Convention** (column "Predictor?"):
- ✅ — used as model input in Step 3
- 🔒 — leaked from target; deliberately excluded
- ⏭️ — deferred to a later Phase
- 🚫 — operational metadata; never a predictor

---

## 0. Outcome (target)

| Column | Type | Fill | Description |
|---|---|---|---|
| `salary_max_usd_yearly` | float, USD/year | 49.8% (6,146 / 12,334) | Maximum disclosed salary, normalized to USD per year (CAD × 0.73; hourly × 2080; monthly × 12). **Trained on `log10` per `LITERATURE_REVIEW.md` §1.1.** Right-skew 2.49 raw → -0.25 log10. Median $195k, P25/P75 $150k / $253k, P99 $485k. |

The companion `salary_min_usd_yearly` (49.8% fill, Spearman 0.93 with the
target) is **excluded** from inputs as label leakage.

---

## 1. Continuous predictors

| Column | Fill | Range / Median | Description | Predictor? |
|---|---|---|---|---|
| `min_years_experience` | 81.8% | 0–30, median 5 | Lowest required years of experience extracted from the description (e.g. "5+ years" → 5, "3-7 years" → 3). Capped at 30 to defang typos. Strongest continuous predictor (Spearman +0.55 with target). | ✅ |
| `max_years_experience` | 4.1% | 0–30 | Upper bound when the description states a range. Sparse — drop in v1. | ⏭️ |
| `max_travel_percent` | 6.5% | 0–100 | Travel requirement as a percentage of time. Near-zero univariate correlation. | ⏭️ |
| `direct_reports_count` | 0.01% | int | Number of direct reports (people-management roles). Almost-empty regex pattern. | ⏭️ |
| `salary_min_usd_yearly` | 49.8% | float, USD/year | Minimum disclosed salary. **Excluded** — paired with the target by ATS, Spearman 0.93 = leakage. | 🔒 |

---

## 2. Ordinal predictors (ordered categories)

| Column | Fill | Levels | Description | Predictor? |
|---|---|---|---|---|
| `min_education` | 25.2% | high_school < associates < bachelors < masters < phd | Minimum required degree extracted from prose. ANOVA F = 105.9 (p ≈ 0). Integer-encode in this natural order; literature also supports years-of-schooling encoding (12, 14, 16, 18, 22) per Mincer 1974. | ✅ |
| `seniority_extracted` | 100% | intern < junior < mid < senior < staff < principal < manager < director < exec | Title-derived seniority via regex; defaults to `mid` when no signal matches (replaced by DeBERTa in Phase 4). Strongest categorical predictor (ANOVA F = 128.6). | ✅ |
| `manager_role` | 25.5% | ic < tech_lead < manager < senior_manager < director < exec | Title-derived management track. Only emitted when title carries a clear signal (no IC default). | ✅ |
| `clearance_level` | 9.8% | public_trust < confidential < secret < top_secret < ts_sci | Required US security clearance. Concentrated at Anduril, SpaceX, Palantir. Integer-encode + missingness indicator. | ✅ |

---

## 3. Nominal predictors — low cardinality

| Column | Fill | Cardinality | Description | Predictor? |
|---|---|---|---|---|
| `country` | 100% | 2 (US, CA) | ISO 3166-1 alpha-2 country code. Stratification key for train/test split. | ✅ |
| `source` | 100% | 3 (greenhouse, lever, ashby) | ATS provider that served the posting. Greenhouse dominates (~86%). Stratification key. | ✅ |
| `role_family_extracted` | 100% | 8 (DS, DA, DE, MLE, RS, AS, SWE-ML, Manager, Other) | Title-derived role family via regex. 'Other' = 70% (non-DS roles like sales / accounting / PM). Replaced by DeBERTa in Phase 4. ANOVA F = 116.0. | ✅ |
| `remote_policy` | 44.7% | 4 (onsite, hybrid, remote, remote-na) | Where the work happens. `remote-na` = remote with NA-only constraint. Promoted from description-text when the location string didn't carry the signal (Step 1a improvement). | ✅ |
| `contract_type` | 34.4% | 5 (full_time, part_time, contract, internship, temporary) | Employment type. ANOVA F = 76.2. | ✅ |
| `equity_form` | 64.9% | 4 (rsu, options, profit_sharing, other) | When equity is offered, what kind. | ✅ |
| `bonus_type` | 5.8% | 4 (signing, annual, performance, retention) | When bonus is mentioned, what kind. Sparse but informative. | ✅ |
| `posting_quality` | 100% | 4 (real, evergreen_pool, talent_community, reposted) | Filter flag for non-genuine postings (e.g. "Future Opportunities — calling all Canadians"). Defaults to `real`. **Used to gate training rows, not as a feature.** | 🚫 → filter |
| `salary_currency` | 49.8% | 2 (USD, CAD) | Original salary currency (already normalized to USD in the target). | 🚫 metadata |
| `salary_period` | 54.5% | 4 (year, month, day, hour) | Original salary period (already annualized). | 🚫 metadata |

---

## 4. Nominal predictors — high cardinality

| Column | Fill | Cardinality | Description | Predictor? |
|---|---|---|---|---|
| `region` | 57.8% | 47 (US states + CA provinces) | 2-letter region code. Target-encode with k-fold CV + Bayesian shrinkage (Pargent 2022). | ✅ |
| `city` | 81.1% | 397 | City name from the location string. Similarity-based (Gamma-Poisson) encoding via `dirty_cat` (Cerda & Varoquaux 2022). | ✅ |
| `company_slug` | 100% | 65 | Internal company identifier. **Risk**: encoding company directly turns the model into "predict company-mean salary" (label leakage at the group level). Recommendation in `LITERATURE_REVIEW.md` §5.2 is to **drop in v1** and let the model learn role-level effects from non-company features. | ⏭️ drop v1 |

---

## 5. Boolean / tri-state predictors

| Column | Fill | Description | Predictor? |
|---|---|---|---|
| `salary_disclosed` | 100% | Whether salary fields are populated. **The target's missingness indicator — never a predictor in the regressor.** | 🔒 selection flag |
| `requires_security_clearance` | 9.8% | True ⇔ posting requires an active US security clearance. Concentrated at defense employers. | ✅ |
| `requires_citizenship` (list) | 14.1% | List of required citizenships, e.g. `["US"]`. Multi-hot encode. | ✅ |
| `offers_visa_sponsorship` | 0.3% | Tri-state (`yes` / `no` / `unspecified`). Sparse; one-hot encode. | ✅ |
| `offers_relocation` | 5.0% | True ⇔ relocation assistance offered. | ✅ |
| `offers_equity` | 64.9% | True ⇔ equity is part of comp. **Strongest boolean predictor** — Δ median = +$54k, p < 1e-300. | ✅ |
| `bonus_mentioned` | 36.3% | True ⇔ a bonus is mentioned. | ✅ |
| `on_call_required` | 4.3% | True ⇔ on-call rotation expected. Sparse but Δ median = +$24k. | ✅ |

---

## 6. List predictors

| Column | Fill | Cardinality (union) | Description | Predictor? |
|---|---|---|---|---|
| `requires_citizenship` | 14.1% | {US, CA} | (Already listed under §5; multi-hot to 2 indicator columns.) | ✅ |
| `language_requirements` | 2.2% | {en, fr, es, ja, zh} | Multi-hot. Sparse but flags QC-French and Japanese postings that slipped the Quebec filter. | ✅ |
| `tech_stack` | 64.7% | ~70 canonical tokens (Python, AWS, SQL, …) | Top-25 multi-hot + `tech_stack_count` + `has_modern_ml` derived columns (`LITERATURE_REVIEW.md` §7.2-3). | ✅ |
| `industry_experience` | 0% | (LLM-only, dormant) | Empty in v1 because the LLM tier is dormant. | ⏭️ |

---

## 7. Text predictors

| Column | Fill | Description | Predictor? |
|---|---|---|---|
| `title` | 100% | Job title as posted. Used to derive `seniority_extracted` and `role_family_extracted`. **Direct title encoding deferred** — Phase 4 will replace the regex extractors with a DeBERTa-v3-base + LoRA classifier trained on the regression target. | ⏭️ Phase 4 |
| `description_md` | 100% | Full description, HTML→markdown converted (Step 1a). Average ~1,500 words. **Phase 5** will embed this with bge-m3 (1024-dim dense) and concatenate to the tabular feature matrix. | ⏭️ Phase 5 |
| `team_or_department` | 0% | (LLM-only, dormant) | Empty in v1. | ⏭️ |

---

## 8. Datetime fields

| Column | Fill | Description | Predictor? |
|---|---|---|---|
| `posted_at` | 86.1% | UTC timestamp of when the ATS first published the posting. Derive `posted_month` (one-hot 12) + `days_since_posted` for the model. | ✅ derived |
| `scraped_at` | 100% | UTC timestamp of when our extractor saw the row. Operational. | 🚫 |
| `first_seen_at` | 100% | First snapshot containing this `id` (curated layer). Operational. | 🚫 |
| `last_seen_at` | 100% | Most recent snapshot containing this `id`. Operational. | 🚫 |

---

## 9. Identity / metadata (never predictors)

| Column | Description |
|---|---|
| `id` | `sha256(company_slug + url)[:16]` — stable across snapshots. **The train/test split is keyed off this column.** |
| `url` | Canonical apply URL. |
| `company_name` | Display name. |
| `company_slug` | (Already in §4 — referenced here for completeness.) |
| `location_raw` | The raw location string from the ATS. Already parsed into `country` / `region` / `city`. |
| `salary_min`, `salary_max` | Original-currency salary values. Normalized into `salary_*_usd_yearly`. |
| `raw_payload_hash` | sha256 of the raw extractor response. Used to detect upstream schema changes. |
| `extraction_meta` | Per-feature provenance JSON (`{"min_years_experience": {"source": "regex", "confidence": 0.85, "rule_id": "years_pattern"}}`). For audit, not modelling. |
| `extraction_version` | Pinned to `"v1"`. Bump when extractors change semantically. |
| `times_seen` | Number of snapshots this `id` has appeared in. Useful for drift tracking. |

---

## 10. Inclusion summary

**Goes into the Step 3 regressor (counted distinct columns):**

| Group | Count | Names |
|---|---|---|
| Continuous | 1 | `min_years_experience` |
| Ordinal | 4 | `min_education`, `seniority_extracted`, `manager_role`, `clearance_level` |
| Low-card nominal | 7 | `country`, `source`, `role_family_extracted`, `remote_policy`, `contract_type`, `equity_form`, `bonus_type` |
| High-card nominal | 2 | `region`, `city` (target-encoded) |
| Boolean / tri-state | 7 | `requires_security_clearance`, `offers_visa_sponsorship`, `offers_relocation`, `offers_equity`, `bonus_mentioned`, `on_call_required`, plus `requires_citizenship` (multi-hot) |
| List | 2 | `language_requirements`, `tech_stack` |
| Datetime-derived | 2 | `posted_month` (one-hot), `days_since_posted` |
| **Total tabular features** | **~60-80 after one-hot / multi-hot expansion** | |

**Deferred / excluded:**

- **Excluded as label leakage** (🔒): `salary_min_usd_yearly`, `salary_disclosed`.
- **Deferred to Phase 4** (DeBERTa replaces regex): `title` direct encoding,
  `seniority_extracted` and `role_family_extracted` (still kept as features
  in v1; replaced under-the-hood when the classifier ships).
- **Deferred to Phase 5** (bge-m3 embedding): `description_md`.
- **Deferred — empty without LLM tier**: `industry_experience`,
  `team_or_department`.
- **Sparse, dropped in v1**: `max_years_experience`, `max_travel_percent`,
  `direct_reports_count`.
- **Used as filter, not feature**: `posting_quality` (drop rows where
  ≠ `real` before training).
- **Used as split keys, not features**: `id` (split hash), `country`,
  `source` (stratification — but they ARE also features; just additionally
  used for stratification).
- **Drop in v1, target-encode revisit later**: `company_slug` (group-level
  leakage risk).

---

## 11. Schema versioning

The data dictionary describes the schema at **`extraction_version="v1"`**.
When extractors meaningfully change (new regex, LLM tier flipped on,
DeBERTa replaces title regex), the version bumps and a new dictionary
revision lands in this file's changelog.

---

## 12. Phase 4 enriched columns (`curated_enriched/jobs.parquet`)

After `curated/enrich.py` runs the trained models over the curated table,
the output parquet at `curated_enriched/jobs.parquet` carries the
49 base columns plus 7 versioned prediction columns. Versioning lets
older predictions remain readable when models retrain (CLAUDE.md §6).

| Column | Type | Source model | Notes |
|---|---|---|---|
| `seniority_label_v1` | string | `arjun10g/na-tech-jobs-seniority-v1` | One of `intern / junior / senior / staff / principal / manager / director`. Trained on regex-confident labels (`"mid"` fallback dropped); val f1_macro 0.831. |
| `seniority_confidence_v1` | float | same | softmax-max from the LR head; useful for filtering |
| `role_family_v1` | string | `arjun10g/na-tech-jobs-role_family-v1` | One of `AS / DA / DE / DS / MLE / RS / SWE-ML`. Trained on regex-confident labels (`"Other"` and `"Manager"` dropped); val f1_macro 0.915. |
| `role_family_confidence_v1` | float | same | softmax-max from the LR head |
| `predicted_salary_usd_v1` | float | `arjun10g/na-tech-jobs-salary-v1` | XGBoost prediction, USD/year. Predicts on every row including non-disclosing ones — see Phase 2 model card for the bias framing. |
| `extracted_skills_v1` | list[string] | `arjun10g/na-tech-jobs-skills-v1` | NuExtract zero-shot skills, normalized to the project taxonomy. **Empty list in v1** — batch enrichment deferred to v1.1 (NuExtract on MPS is 6 hours for 12k rows). |
| `prediction_model_version` | string | — | Currently `"v1"`. Bumps when any of the four models retrains. |

---

## 13. Changelog

- **2026-05-08 (v1, Phase 4 enrichment)**: added the 7 versioned
  prediction columns above. Underlying models: frozen-MiniLM + LR
  (seniority, role_family), XGBoost (salary), NuExtract zero-shot
  (skills, batch deferred to v1.1).
- **2026-05-08 (v1)**: First draft. Covers all 49 columns of the
  2026-05-08 curated snapshot. Inclusion decisions match
  `LITERATURE_REVIEW.md` §14 recommendations table.
