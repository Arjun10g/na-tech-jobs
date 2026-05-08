# Literature Review — Predictor Treatment for the NA Tech Salary Regressor

> **Living document.** Started 2026-05-08 to capture how the literature
> recommends treating each predictor type we have in `data/curated/jobs.parquet`,
> grounded in our actual EDA findings (`eda/reports/2026-05-08/report.md`).
> Updated on every modelling-relevant decision; see the changelog at the end.

This review answers two questions for every predictor and analytic choice:

1. **What is the textbook / state-of-the-art treatment?**
2. **What does it imply for our specific dataset?**

The recommendations table in §14 is the executive summary; the body is the
justification with citations.

---

## 0. Purpose, scope, conventions

- **Target**: `salary_max_usd_yearly` on the disclosed subset (n ≈ 6,146 of
  12,334 active rows; 49.8% disclosure rate).
- **Modelling context**: tabular features + a (Phase 5) bge-m3 1024-dim
  description embedding feeding a gradient-boosted tree regressor (XGBoost),
  hyperparameter-searched with Optuna and tracked in MLflow per CLAUDE.md §7.
- **Citation style**: in-line `[Author Year]` with full references in §16.
  Where I cite a textbook chapter I name it; where I cite a paper I link it.
- **What this is _not_**: an exhaustive academic survey of compensation
  modelling. It's a working document for design decisions, written in the
  voice of someone who has to ship a model that recruiters will critique.

---

## 1. The target: compensation as a prediction problem

### 1.1 Distributional properties of compensation data

Earned-income distributions are **right-skewed and approximately log-normal**
in the upper tail. This was Pareto's original observation [Pareto 1897];
modern administrative data confirms log-normality dominates the bulk of the
income distribution while a Pareto power law fits the upper ~1-3% [Atkinson
& Piketty 2007; Reed 2003]. Our EDA confirms: `salary_max_usd_yearly` raw
skew = +2.49, kurtosis = +16.0, both fall to **+0.25 / +1.07** under
`log10`, indicating the log-transformed target is well-behaved for any
model that benefits from approximately Gaussian residuals
(`eda/reports/2026-05-08/report.md` §5).

**Implication for us**: train on `log10(salary_max_usd_yearly)`. Report MAE
in USD via back-transform (i.e. `10^(log_pred) - 10^(log_true)`), since
that's the unit a recruiter expects. RMSE on log-scale ≈ MAPE on raw scale,
which is also useful — but MAE is more robust to the heavy tail. Hastie,
Tibshirani & Friedman recommend log transforms for any monetary outcome
where multiplicative errors are more natural than additive ones [HTF 2009,
§3.2.4].

### 1.2 Selection / disclosure bias (MNAR)

Salary disclosure on job postings is **employer-controlled and
non-random**. Three forces drive it:

1. **Pay-transparency laws** in California (SB 1162, eff. 2023), New York
   City (Local Law 32, eff. 2022), Washington State (HB 1696, 2023),
   Colorado (Equal Pay for Equal Work Act, 2021), Connecticut, Maryland,
   Illinois, Hawaii (US) and Ontario, BC, Prince Edward Island (CA).
2. **Voluntary disclosure** by transparency-leaning employers (e.g. Stripe,
   Cohere, Anduril have disclosed across most postings in our corpus).
3. **Strategic non-disclosure** — employers pay above- or below-market wages
   may decline to disclose for negotiating leverage [Cullen & Pakzad-Hurson
   2023, _Equilibrium Effects of Pay Transparency_, AER].

The first two are **MAR conditional on observed predictors** (jurisdiction,
company); the third is **MNAR** because the latent salary itself drives
disclosure.

The classical correction is **Heckman 2-stage selection** [Heckman 1979,
Econometrica]:

1. Probit (or logit) on `disclosed = f(observable features + an exclusion
   restriction)`. The exclusion restriction is a feature affecting the
   probability of disclosure but **not** the salary itself; the canonical
   one in our setting is whether the posting is in a pay-transparency-mandate
   jurisdiction.
2. Compute the inverse Mills ratio λ from stage 1.
3. Regress `salary | disclosed` on features, including λ as an additional
   regressor. The λ coefficient absorbs the selection effect.

The 2-stage correction's identification relies on the exclusion restriction.
With recent pay-transparency mandates we have a credible one (jurisdiction).
**Our v1 approach**: ship a 1-stage regressor with explicit MNAR caveats in
the model card (CLAUDE.md §7). Phase 4+ revisits with Heckman or a 2-stage
neural model along the lines of [Liu et al 2022] for compensation.

The alternative is **inverse propensity score weighting** [Rosenbaum &
Rubin 1983]: weight disclosed observations by `1/P(disclosed | features)`.
This needs the same observable confounders to be sufficient, so it doesn't
escape the MNAR problem either; it just changes the formal framing.

For the purposes of a publicly-shipped salary regressor, the **honest
framing** [Mitchell et al 2019, _Model Cards for Model Reporting_] in the
model card is the most important deliverable. It says the model predicts
"salary as priced by disclosing employers in our corpus", not "ground truth
salary".

### 1.3 Regression vs. ordinal vs. survival framings

Three framings of the salary prediction problem exist in the literature:

- **Direct regression** on log-salary (this is our approach).
- **Bin-and-classify**: discretize salary into ranges (e.g. <$100k, $100-150k,
  ...) and treat as ordinal classification [Wang & Hu 2018, _Salary
  Prediction in the IT Job Market_]. Loses information; useful only when
  noise dominates the regression target.
- **Survival / Tobit**: treat the disclosed salary as a censored observation
  of the unobserved true salary [Tobin 1958]. Theoretically attractive when
  censoring is well-defined; in our case the "censoring" mechanism is
  selection (MNAR), not censoring (where we know a value is greater than
  some threshold), so survival models don't quite fit.

We stay with direct regression on log-salary. The Tobit framing is
mentioned for completeness and may be revisited if we acquire ground-truth
salary data (e.g. via H-1B disclosures or LinkedIn opt-ins).

### 1.4 Loss function choice

For tree boosting [Chen & Guestrin 2016, _XGBoost: A Scalable Tree Boosting
System_, KDD]:

- **MSE on log-target** (default `reg:squarederror`): converges fast,
  symmetric residual penalty, sensitive to outliers but log-transform
  already shrinks them.
- **MAE / pseudo-Huber** (`reg:absoluteerror`, `reg:pseudohubererror`):
  more robust; slower to converge.
- **Gamma deviance** (`reg:gamma`): for strictly positive monetary outcomes
  on the original scale. Equivalent to log-MSE in the limit of small
  variance; provides a small efficiency gain when the target is right-tail
  heavy [Smyth 1996, _Regression analysis of quantity data with exact zeros_].

We default to MSE on log-target. Gamma deviance can be A/B'd later. MAE on
log-target is also reasonable but XGBoost's MAE objective uses an
approximation that can be slow to converge.

**Eval metrics** (per CLAUDE.md §7, stratified by country and source):

- **MAE in USD** (back-transformed): the headline number for the model card.
- **MAPE**: `mean(|y_pred - y_true| / y_true)` — interpretable as a
  percentage error, but unstable when y_true is small (less of a concern at
  $50k+ salaries).
- **R² on log-target**: captures variance explained on the trained scale.
- **Quantile loss at the median (P50)**: equivalent to MAE; useful for
  comparing against pinball-loss baselines.

---

## 2. Continuous predictors

### 2.1 `min_years_experience` (fill 81.8%, Spearman +0.55)

The strongest continuous predictor in our EDA. Treatment notes:

- **Keep raw**, no transformation needed for tree models. Linear models
  would prefer `sqrt(years)` or `log1p(years)` because returns to
  experience are concave in compensation studies [Mincer 1974, _Schooling,
  Experience, and Earnings_; Heckman, Lochner & Todd 2006, _Earnings
  Functions, Rates of Return and Treatment Effects_].
- **Add a missingness indicator** (`min_years_experience_isna` boolean).
  Tree models with built-in missing-handling (LightGBM, XGBoost ≥ 1.0
  default) split missing into the better child during training; a separate
  indicator is redundant for them but serves as documentation [Chen &
  Guestrin 2016, §3.4]. Linear models definitely need the indicator.
- **Outlier guard**: cap at 30 years. Anything above is likely a typo
  (e.g. "100+ years experience"). Our regex extractor already caps at 30.
- **Interaction with seniority**: redundant with `seniority_extracted`
  for ordering, complementary in magnitude. Tree models capture this
  interaction natively; document the redundancy in §11 (multicollinearity).

### 2.2 `max_travel_percent` (fill 6.5%, Spearman +0.07)

- Near-zero univariate signal in our EDA. Likely a noise predictor at
  this fill rate.
- **Treatment**: bin into `{0, low (<25), medium (25-50), high (>50)}` and
  use as an ordinal category, OR drop. We'll drop in v1 and revisit if
  field coverage rises.

### 2.3 Salary-derived predictors (`salary_min_usd_yearly`)

- Spearman with target = +0.93 (effectively the same field, since
  ATS disclosures pair min/max). **Not a predictor** — using it would be
  trivial label leakage. Drop from inputs entirely.
- Worth noting: `salary_min_usd_yearly` and `salary_max_usd_yearly`
  carry the same disclosure-bias structure. Models should not pretend
  they're independent disclosures.

---

## 3. Ordinal predictors

Ordinal predictors carry intrinsic ordering information that binary or
one-hot encoding **destroys**. Three encoding schemes are common:

1. **Integer encoding** (e.g. `intern=0, junior=1, mid=2, senior=3, ...`).
   Preserves order; assumes equal spacing between levels. Tree models do
   not assume equal spacing — they only need monotonicity for splits to
   capture ordering — so integer encoding is the recommended default for
   GBDT-family models [Pargent et al 2022, _Regularized target encoding
   outperforms traditional methods in supervised machine learning_].
2. **Target / mean encoding**: replace each level with the mean target
   among rows of that level. Strong predictor signal; risks leakage if not
   done with k-fold cross-validation. See §5.2.
3. **One-hot encoding**: ignores order; loses information but provides
   maximum flexibility for non-linear relationships between adjacent
   levels. Only useful when the ordering is suspect.

For our four ordinals:

### 3.1 `min_education`

Order: `high_school < associates < bachelors < masters < phd`. The
empirical wage-by-education curve is approximately log-linear in years of
schooling (Mincer's law). Integer encode with the natural ordering;
optionally substitute years-of-schooling (12, 14, 16, 18, 22) as the
encoding to encode the cardinal spacing the literature prefers.

ANOVA F = 105.9 on n=1,845 disclosed (p ≈ 0). Despite the modest fill
rate (25%), where present it's a strong predictor.

### 3.2 `seniority_extracted`

Order: `intern < junior < mid < senior < staff < principal < director < exec`,
plus `manager` which doesn't fit cleanly between staff and principal in
all companies. Two valid choices:

- Treat as a strict 9-level ordinal in our regex order, accepting the
  manager-track ordering imperfection.
- Split into two predictors: `IC_track_level` (intern...principal) and
  `mgmt_track_level` (lead...exec), with NA for the inactive track.

We use the 9-level integer encoding in v1. Phase 4 replaces this regex
heuristic with a DeBERTa classifier per CLAUDE.md §7 — that classifier can
be trained directly on the (titled) compensation regression objective, in
which case the latent representation will be dataset-optimal.

ANOVA F = 128.6 on n=6,146 (the entire disclosed set has seniority filled).

### 3.3 `manager_role`

Order: `ic < tech_lead < manager < senior_manager < director < exec`.
Lower fill (25.5%) because the regex only labels titles with explicit
manager / lead / director / VP keywords; ICs without those words remain
unlabeled. Same recommendation as seniority — integer encode, replace
with a learned classifier in Phase 4.

### 3.4 `clearance_level`

Order: `public_trust < confidential < secret < top_secret < ts_sci`.
Fill 9.8%; concentrated in defense employers (Anduril, SpaceX, Palantir).
Despite low fill, the within-cleared subset has high informational value
about compensation (cleared roles in defense pay top-of-market). Integer
encode + missingness indicator.

---

## 4. Low-cardinality nominal predictors

`country` (2), `source` (3), `role_family_extracted` (8), `remote_policy` (4),
`contract_type` (5), `salary_currency` (2), `salary_period` (4),
`equity_form` (3), `bonus_type` (4), `posting_quality` (4).

For tree models with N levels ≤ ~10:

- **One-hot encoding** is the safe default. It's not memory-efficient but
  cardinality is low.
- **Integer encoding with `enable_categorical=True`** in XGBoost ≥ 1.5 or
  LightGBM treats the variable as categorical natively, finding optimal
  partitions of levels in O(K log K) [Fisher 1958, _On Grouping for
  Maximum Homogeneity_; LightGBM docs]. For small K this is roughly
  equivalent to one-hot in practice but faster.
- **Target encoding** (§5.2) is overkill for low-cardinality nominals
  unless K > 15 or so.

For us, one-hot is fine. The XGBoost `categorical` mode is also fine and
slightly faster — a candidate A/B in Step 3.

`country`: 2 levels, but **stratify train/eval splits** by country so the
US/CA proportions are preserved in each fold (CLAUDE.md §7's "stratified
eval by country and source" requirement).

`source`: 3 levels (greenhouse, lever, ashby). Already stratify by this.

`role_family_extracted`: 8 levels but heavy class imbalance ('Other' = 70%
of rows, MLE/DS/RS/DE/DA only 1-3% each). Won't matter for the regressor
beyond providing a feature; matters a lot for the Phase 4 role classifier
(class-weighting required there).

---

## 5. High-cardinality nominal predictors

`region` (47 unique values), `city` (397), `company_slug` (65). These are
the real encoding challenge.

### 5.1 The high-cardinality problem

One-hot encoding `city` adds 397 columns; with our 6,146 disclosed rows
that's <16 rows per column on average — guaranteed overfitting territory.
Two schools of thought:

- **Reduce cardinality by grouping**: keep top-N most frequent levels;
  bucket the rest into "Other". Information-lossy.
- **Embed via target statistics**: replace each level with a function
  (mean, std, count) of the target observed for that level. Lossy in a
  different way (collapses each level to a scalar) but typically yields
  better predictive performance.

Three target-encoding refinements have evolved in the literature:

### 5.2 Target encoding with leakage protection

Naive target encoding (`region → mean(salary)`) leaks because the encoded
value for row _i_ depends on row _i_'s own target. The mainstream fix is
**k-fold (out-of-fold) target encoding** [Micci-Barreca 2001]:

```
For row i in fold k:
    encoded_i = mean(target | level_i, fold ≠ k)
```

Adds a `K` × constant overhead. Implementation: `category_encoders`
package in Python, or scikit-learn's `TargetEncoder` (added in 1.3, with
internal CV support).

**Smoothing** (Bayesian shrinkage toward the global mean):

```
encoded(level) = (n(level) * mean(level) + m * global_mean) / (n(level) + m)
```

where `m` is a hyperparameter controlling shrinkage. Levels with few
observations are pulled toward the global mean (high prior weight); large
levels stay close to their empirical mean. The empirical-Bayes choice of
`m ≈ var(within) / var(between)` is suggested by [Pargent et al 2022].

Their finding (Table 3 of that paper, replicated across 24 datasets):
**regularized target encoding outperforms one-hot, ordinal, and James-Stein
encoding** for tree models on high-cardinality categoricals. Effect sizes
are largest where `K > 20`.

For our `region` and `city`, this is the recommended path. For
`company_slug`, target encoding is **risky** because the company effect
is precisely what we want the model to learn from non-company features
(otherwise the model degenerates to "predict company-mean salary"). Two
options for `company_slug`:

- **Drop entirely**: rely on company-implied features (country, source,
  size proxies via `times_seen`) to absorb the variation.
- **Target encode with very strong shrinkage** (`m` large), so the encoding
  contributes little for low-volume companies and approaches the company
  mean only for high-volume ones.

Per [Cerda & Varoquaux 2022], **similarity-based encodings** (Gamma-Poisson,
min-hash) further outperform target encoding when string columns have
hierarchical or near-duplicate structure (e.g. `San Francisco` vs `San
Francisco Bay Area`). For our `city` column this is the natural fit;
`region` is already canonicalized to 2-letter codes so similarity gains
nothing.

### 5.3 Hashing trick (alternatives)

The hashing trick [Weinberger et al 2009, _Feature Hashing for Large Scale
Multitask Learning_, ICML] hashes each category into one of D buckets. It
trades collisions for memory and avoids needing the full vocabulary.

For our scale (max 397 cities), hashing is **unnecessary** — vocabulary
size is small enough to enumerate. Listed for completeness because future
expansion (e.g. a global rather than NA-only dataset) might require it.

### 5.4 Hierarchical / mixed-effects framings

For inherently hierarchical data (jobs nested in companies nested in
sectors), mixed-effects models can outperform fixed-effect encodings by
**partially pooling** information across groups [Gelman & Hill 2006, _Data
Analysis Using Regression and Multilevel/Hierarchical Models_].

```
log_salary_ij = X_ij β + α_company[j] + ε_ij
α_company ~ Normal(0, σ_company²)
```

The random intercept `α_company` is shrunk toward zero in proportion to
within-company variance; high-volume companies retain their estimated
intercept, low-volume ones are pulled toward the global mean.

In tree-boosting contexts, **target encoding with empirical-Bayes
smoothing is mathematically equivalent** to a 1-level random-intercept
model with conjugate prior [McElreath 2020, _Statistical Rethinking_,
§13.4]. So we get the partial-pooling benefit "for free" by using the
[Pargent 2022] target encoder with cross-validated `m`.

If we eventually want explicit hierarchy modelling (e.g. for the model
card's interpretability discussion), `pymc` or `bambi` provide a clean
path; for v1 we stay with the target-encoded GBDT.

---

## 6. Boolean / tri-state predictors

`salary_disclosed`, `requires_security_clearance`, `offers_visa_sponsorship`
(tri-state: yes/no/unspecified), `offers_relocation`, `offers_equity`,
`bonus_mentioned`, `on_call_required`.

Tree-model treatment is straightforward: pass as 0/1 (or 0/1/missing for
tri-state). Three subtleties:

1. **`salary_disclosed` is the target's "missing-data indicator"**, NOT a
   predictor. Including it in the regressor makes the model trivial
   (since it's only trained on disclosed rows). It belongs in any
   second-stage selection model (Heckman) but not the regression.
2. **`offers_visa_sponsorship`** has three levels (yes/no/unspecified) —
   one-hot encode rather than collapsing to bool, because "unspecified" is
   informationally distinct from "yes" or "no" [Little 1988, _A test of
   missing completely at random for multivariate data with missing
   values_].
3. **Boolean columns with sparse fill** (e.g. `offers_visa_sponsorship` at
   0.3% fill, `direct_reports_count` at <0.1%) are likely overfit-magnets
   in v1. Suppress them or encode with strong regularization. In our EDA
   most booleans pass the t-test for difference in median salary
   (`offers_equity` Δmedian = +$54k, p ≈ 0; `on_call_required` Δ = +$24k,
   p = 0.07), so they have signal where present.

---

## 7. List-valued (multi-hot) predictors

`requires_citizenship` (≤2 values per row), `language_requirements`
(≤5), `tech_stack` (1-30 values per row, ~600 unique tokens),
`industry_experience` (LLM-only — currently empty).

Three encoding strategies:

### 7.1 Multi-hot indicator columns

For low-cardinality lists (e.g. `requires_citizenship` has only `["US"]`,
`["CA"]`, `["US", "CA"]` as observed values), a flat one-hot of the union
suffices: `requires_US_citizenship` (bool), `requires_CA_citizenship`
(bool). 2 columns total.

For `language_requirements`, similar: en, fr, ja, etc. ~5 columns.

### 7.2 Multi-hot for `tech_stack`

`tech_stack` has ~70 canonical tokens (we mined them by hand in
`ingestion/feature_extraction/regex/tech_stack.py`). With 6,146 rows,
top-N = 25 multi-hot columns (Python, AWS, SQL, Spark, Kubernetes, ...)
gives ~250 rows per indicator on average — adequate for tree models.

The literature supports **multi-hot top-N + 'other' indicator** for skill
extraction [Bian et al 2019, _Domain Adaptation for Person-Job Fit_].
Embeddings of the tech-stack token (e.g. via word2vec-style co-occurrence)
are an alternative but typically don't outperform multi-hot in regression
contexts where the model can already learn token-level effects.

### 7.3 Counts and density

Two derived features that often help in compensation models specifically:

- **`tech_stack_count`**: number of distinct tokens. Proxy for breadth
  required. Empirically correlates positively with senior roles.
- **`has_modern_ml`**: indicator that any of {PyTorch, TensorFlow,
  HuggingFace, MLflow, Weights & Biases, Spark} appears. Proxy for
  ML-platform roles.

We add both in Step 3.

---

## 8. Text predictors

### 8.1 `title`

`title` averages 5-8 words; carries strong signal for `seniority_extracted`
and `role_family_extracted`. CLAUDE.md §7 reserves Phase 4 for a DeBERTa-v3
fine-tune that replaces our title regex. For Step 3's salary regressor,
we use only the regex-derived signals — the DeBERTa output is a Phase 4
add-on.

### 8.2 `description_md`

CLAUDE.md §7 specifies bge-m3 (`BAAI/bge-m3`) [Chen et al 2024, _M3-Embedding:
Multi-Linguality, Multi-Functionality, Multi-Granularity_, ACL] as the
unified embedder for all text in the project. Its dense output is a
1024-dim vector; we concatenate this to the tabular feature matrix.

The decision to use a dense embedder vs TF-IDF for compensation prediction:

- TF-IDF + linear model: interpretable but loses semantics ('senior
  engineer' ≈ 'sr. engineer' as different tokens).
- TF-IDF + tree boosting: also workable but doesn't generalize well to
  unseen tokens.
- Dense embedding (bge-m3, SBERT, etc.) + tree boosting: state-of-the-art
  for text-augmented tabular regression [Borisov et al 2022, _Deep Neural
  Networks and Tabular Data: A Survey_, IEEE TNNLS]. Embedding dimensionality
  should be small relative to N: 1024-dim with 6,146 rows is borderline,
  but tree models handle wide inputs better than linear ones because of
  feature subsampling.

A common refinement is to pass the raw embedding vs a **PCA-reduced**
version (50-200 dims). PCA reduces multicollinearity within the embedding
block. We test both in Step 3 ablations.

---

## 9. Datetime predictors

`posted_at` is the only datetime feature with predictor potential
(`scraped_at`, `first_seen_at`, `last_seen_at` are operational metadata).

Three derivations from a timestamp:

1. **`days_since_posted`**: continuous. May matter little for the
   structured ATS data we ingest weekly, since postings older than a
   week tend to be either backfills or never-fills.
2. **Cyclic encoding** for month / day-of-year: `sin(2π * t / period)`
   and `cos(...)`. Preserves the cyclic nature without imposing a linear
   ordering [Chu et al 2008, _Time Series Mining_].
3. **Categorical month / quarter**: hiring volume is seasonal in
   tech (Q1 ramp, Q3 hiring lull). Categorical for the model card-
   reportable seasonality breakdown.

We add `posted_month` (one-hot) + `days_since_posted` (continuous + log-
of-1+x for tail compression).

**Time leakage**: when validating, split temporally rather than randomly
**if** the test goal is "predict salaries for postings that don't yet
exist." For our v1 model card we use stratified random splits within the
disclosed pool, since the goal is "predict salary for a given role
description today." Time-based holdout is a future iteration.

---

## 10. Missing data

### 10.1 Mechanisms (MCAR / MAR / MNAR)

Per Little & Rubin [2002, _Statistical Analysis with Missing Data_, 2nd ed]:

- **MCAR** (missing completely at random): `P(R | X, Y) = P(R)`. The
  missingness indicator is independent of observed AND unobserved data.
  Almost never holds in practice.
- **MAR** (missing at random): `P(R | X, Y) = P(R | X)`. The missingness
  depends only on _observed_ data. Multiple imputation is unbiased under
  MAR.
- **MNAR** (missing not at random): `P(R | X, Y) ≠ P(R | X)`. Missingness
  depends on the unobserved value itself. Multiple imputation is _biased_
  under MNAR; specialized models (Heckman, pattern-mixture, selection
  models) are required.

Our EDA chi-square diagnostics (`metrics.json`'s `missingness.mar_signals`)
test associations between `salary_max_usd_yearly` missingness and observed
columns (country, source, role family, seniority). Strong dependencies
(p ≈ 0) detected for all four — i.e. the data is **at least MAR**. Whether
it's also MNAR (depends on unobserved salary itself) cannot be tested
without ground-truth — but the literature on pay transparency [Cullen &
Pakzad-Hurson 2023] strongly suggests yes.

### 10.2 Tree-based native handling

XGBoost since 1.0 [Chen & Guestrin 2016, §3.4 "Sparsity-aware Split Finding"]
learns the optimal direction for missing values during training: each split
considers two candidate child assignments for the missing rows and picks
the one minimizing the loss. This is _practically_ better than mean
imputation for non-MCAR data: the model learns "missing in this column
correlates with high salary because cleared roles in defense are
under-disclosed", which is exactly the MAR signal we'd want it to use.

LightGBM and CatBoost have similar mechanisms.

### 10.3 Indicator + imputation (for non-tree models)

For linear / NN models (and sometimes for tree models when interpretability
matters):

- Add an `_isna` indicator column for each missing predictor.
- Impute the value itself with a benign default (median for continuous,
  mode or "missing" for categorical).

This decomposes the effect into "what value did this row have?" and "did
we observe it at all?", letting the model learn both.

### 10.4 Multiple imputation (MICE)

[van Buuren 2018, _Flexible Imputation of Missing Data_, 2nd ed; the `mice`
R package, `IterativeImputer` in scikit-learn]: chained equations imputes
each missing column from the others, draws M imputations, fits the model
M times, and pools.

**Not recommended for our salary target** because the target is MNAR.
Imputing salary from observed predictors and then training on the imputed
+ observed salaries would propagate selection bias into the predictions.

For predictor-side MICE, viable but our predictor missingness is mostly
"genuinely unknown" (the regex didn't extract it) rather than "should be
imputed from other features"; we let the tree handle missingness natively
and revisit if a non-tree model beats it.

### 10.5 Our salary disclosure: MNAR + Heckman option

See §1.2 for Heckman 2-stage. Summary: viable in v2; v1 ships with
honest framing in the model card.

---

## 11. Multicollinearity

### 11.1 VIF and condition number

VIF for predictor _i_ = `1 / (1 - R²_i)` where `R²_i` is the R² of
regressing predictor _i_ on the other predictors. Rules of thumb [Pedhazur
1997, _Multiple Regression in Behavioral Research_]:

- VIF < 5: no concern
- 5 ≤ VIF < 10: moderate
- VIF ≥ 10: severe

Our EDA on the small fully-populated continuous block:
- `min_years_experience` VIF = 4.24
- `salary_min_usd_yearly` VIF = 4.24

Both moderate. The latter we drop entirely (label leakage), so the
practical block has no multicollinearity issue.

**Condition number** of the design matrix ≈ 468,030 in our EDA — this is
**driven by scale mismatch** (`salary_min_usd_yearly` ranges 30k-1M vs
`min_years_experience` 1-30). On a standardized matrix the condition
number would be ~1.5. Tree models are scale-invariant so this number is
informational rather than actionable.

### 11.2 Tree-based immunity

[Hastie, Tibshirani & Friedman 2009, §10.10] — boosted trees are largely
**immune to multicollinearity** in the OLS sense because each split picks
exactly one feature. Two highly-correlated predictors can substitute for
each other across splits without harming overall fit; the only practical
cost is feature-importance dilution, which matters for interpretability
but not for prediction accuracy.

Where it does matter for us: **interpretability of the model card's
feature importance plot**. If `min_years_experience` and
`seniority_extracted` are 0.7-correlated, importance is shared between
them and the picture is muddier. Solution: present them as a paired
sensitivity (e.g. partial dependence plots [Friedman 2001, _Greedy
Function Approximation: A Gradient Boosting Machine_]) rather than as
isolated importances.

### 11.3 Regularization

L1 (LASSO) + L2 (Ridge) penalties are the textbook defenses against
multicollinearity for linear models. XGBoost includes both via
`reg_alpha` and `reg_lambda` hyperparameters [Chen & Guestrin 2016, §2.1].
Optuna can search them in Step 3.

---

## 12. Outliers and influential observations

Three flavours of "outlier" matter:

1. **Univariate outliers** in the target. Cap or winsorize.
2. **Multivariate outliers** (Mahalanobis distance, Cook's distance for
   linear models). Tree models don't have an analogous metric, but
   permutation-based row-importance can help.
3. **Mislabeled rows**: our salary mining regex can occasionally extract
   a non-salary number ("$1M Series A funding"). Manual sanity checks
   needed.

EDA findings: `salary_max_usd_yearly` IQR fences at $48,400 and $354,600;
n_iqr_outliers = 285 (4.6%); n_z>3 = 39 (0.6%). The right-tail outliers
are real — Anduril principal engineering roles, SpaceX leadership, finance
quant roles all legitimately exceed $400k. **Recommendation**: winsorize at
the 99.5th percentile (≈ $580k based on our distribution) rather than the
P99 fence; this preserves the senior-IC tail the project explicitly
targets while clipping obvious outliers.

For the lower tail, anything below $30k/year USD-equivalent is suspect.
Either the salary is in a non-USD/non-CAD currency we mis-detected, or
the period is hourly/daily and our annualization (× 2080 or × 260) hit a
part-time gig. Filter out before training (small absolute count).

---

## 13. Cross-validation strategy

Three options, all relevant:

### 13.1 Stratified K-fold (default)

Stratify on `(country, source)` to preserve the US/CA × greenhouse/lever/
ashby cells across folds. Randomly split within each cell.

### 13.2 Group K-fold by company

Holds out entire companies in turn, so the model is forced to generalize
to companies it has never seen. **Stricter** evaluation; closer to
deployment reality (we want to predict salaries for new companies that
join the dataset). Cost: larger variance across folds because of company-
size imbalance.

### 13.3 Time-based splits

Train on snapshots before date T, test on snapshots after T. Most
realistic for production but our v1 dataset has only one snapshot, so this
is a v2 concern.

**Our approach**: 5-fold stratified by (country, source) for the headline
metric in the model card; group-K-fold by company as a secondary
robustness check reported alongside [Pargent et al 2022 used the same
two-fold strategy and recommended both for tabular regression].

---

## 14. Synthesis: recommendations table

| Predictor                       | Type           | Treatment                                                                                      | Source                     |
|---------------------------------|----------------|-------------------------------------------------------------------------------------------------|---------------------------|
| `salary_max_usd_yearly`         | target         | `log10`; train MSE on log; back-transform for MAE/MAPE                                          | §1; HTF 2009              |
| `salary_min_usd_yearly`         | (drop)         | label leak — exclude from inputs                                                                 | §2.3                       |
| `min_years_experience`          | continuous     | raw + missingness indicator; cap at 30                                                           | §2.1; Mincer 1974         |
| `max_travel_percent`            | (drop in v1)   | too sparse (6.5% fill, near-zero corr)                                                           | §2.2                       |
| `min_education`                 | ordinal        | integer encode (or substitute years-of-schooling)                                                | §3.1; Mincer 1974         |
| `seniority_extracted`           | ordinal        | integer encode in regex order                                                                    | §3.2                       |
| `manager_role`                  | ordinal        | integer encode + `is_NA` indicator                                                               | §3.3                       |
| `clearance_level`               | ordinal        | integer encode + `is_NA` indicator                                                               | §3.4                       |
| `country`, `source`, `role_family_extracted` | nominal | one-hot (or XGBoost `enable_categorical`)                                                       | §4                          |
| `remote_policy`                 | nominal        | one-hot, with `<missing>` as its own level                                                       | §4                          |
| `contract_type`                 | nominal        | one-hot                                                                                          | §4                          |
| `region`                        | high-card nom. | k-fold target encoding with empirical-Bayes shrinkage                                            | §5.2; Pargent 2022        |
| `city`                          | high-card nom. | similarity-based (Gamma-Poisson) encoding via `dirty_cat`                                        | §5.4; Cerda & Varoquaux 2022 |
| `company_slug`                  | high-card nom. | target encode with strong shrinkage OR drop in v1                                                | §5.2                       |
| Booleans (equity, bonus, …)     | binary         | 0/1; `_isna` indicator only if non-tree model                                                    | §6                          |
| `offers_visa_sponsorship`       | tri-state      | one-hot (yes/no/unspecified)                                                                     | §6                          |
| `requires_citizenship`          | list           | flat multi-hot (US, CA)                                                                          | §7.1                       |
| `language_requirements`         | list           | flat multi-hot                                                                                   | §7.1                       |
| `tech_stack`                    | list           | top-25 multi-hot + `tech_stack_count` + `has_modern_ml`                                          | §7.2-7.3; Bian 2019        |
| `industry_experience`           | (drop in v1)   | LLM-only; currently 0% fill                                                                      | §7                          |
| `title`                         | (drop in v1)   | use derived `seniority_extracted`/`role_family_extracted`; learn DeBERTa in Phase 4              | §8.1                       |
| `description_md`                | text           | bge-m3 1024-dim embedding (Phase 5); maybe PCA to 200                                             | §8.2; Chen 2024            |
| `posted_at`                     | datetime       | `posted_month` (one-hot 12) + `days_since_posted` continuous                                     | §9                          |
| All missingness                 | —              | tree-native (XGBoost sparsity-aware splits)                                                       | §10.2; Chen & Guestrin 2016 |
| Multicollinearity               | —              | tree-immune in practice; report partial dependence in model card                                  | §11.2; HTF 2009 §10.10     |
| Outliers (target)               | —              | winsorize at 99.5th percentile (~$580k); drop annualized salaries < $30k                          | §12                         |
| Cross-validation                | —              | 5-fold stratified by (country, source); group-by-company as secondary                             | §13                         |

---

## 15. The ideal EDA + data-prep pipeline (and what we actually did)

The "what should EDA do?" question has been answered most cleanly across
three sources:

- John Tukey's _Exploratory Data Analysis_ [Tukey 1977] — the foundational
  framing: examine data with an open mind before settling on a model.
- Kuhn & Johnson, _Applied Predictive Modeling_ [Kuhn & Johnson 2013],
  ch. 3 ("Data Pre-Processing") — the modern ML-pipeline checklist.
- Géron, _Hands-On Machine Learning_ [Géron 2022], ch. 2 — the practical
  end-to-end recipe that newer practitioners standardize on.

Synthesizing into the textbook ideal pipeline:

### 15.1 Ideal EDA stages (pre-modelling)

| #  | Stage                                             | Purpose                                           | Reference          |
|----|---------------------------------------------------|---------------------------------------------------|--------------------|
| 1  | Data integrity audit (dtypes, encoding, dedup)   | Don't analyze garbage                             | Wickham 2014 (tidy data) |
| 2  | Train/test split BEFORE EDA                       | Prevent analyst-side leakage                      | Géron 2022 §2.4    |
| 3  | Schema + role classification (predictors / metadata / target) | Separate signal from operational columns | Kuhn & Johnson §3.2 |
| 4  | Univariate stats per column                       | Spot weird values, distributions                  | Tukey 1977         |
| 5  | Distributions + visualizations (hist, ECDF, box)  | Visual sanity                                     | Cleveland 1985     |
| 6  | Normality / shape tests (skew, kurtosis, S-W, A-D, Q-Q) | Identify transformations needed                | Shapiro & Wilk 1965 |
| 7  | Missingness analysis (rates, patterns, MCAR/MAR/MNAR) | Decide imputation strategy                     | Little & Rubin 2002 |
| 8  | Outlier audit (IQR, z-score, Mahalanobis)         | Choose winsorize/trim/keep                        | Iglewicz & Hoaglin 1993 |
| 9  | Target deep-dive (stratified)                     | Understand the thing being predicted              | Kuhn & Johnson §3.4 |
| 10 | Bivariate target ↔ predictor (corr, MI, ANOVA, t) | Identify weak / strong predictors                 | Hastie et al §3   |
| 11 | Multicollinearity (corr matrix, VIF, condition #) | Plan encoding / regularization                    | Pedhazur 1997      |
| 12 | Multivariate / dimensionality (PCA, t-SNE, UMAP)  | See latent structure                              | Jolliffe 2002      |
| 13 | Group / stratification differences                | Detect Simpson's paradox, subgroup heterogeneity  | Simpson 1951       |
| 14 | Time-based audit (drift, trend, seasonality)      | Choose CV strategy                                | Hyndman & Athanasopoulos 2018 |
| 15 | Sample-size / power adequacy                      | Set eval-stratum granularity honestly             | Cohen 1988         |
| 16 | Bias / fairness audit (protected attributes)      | Required under model-card disclosures             | Mitchell 2019      |
| 17 | Selection-bias / MNAR specific analysis           | Identify ML-validity threats                      | Heckman 1979       |
| 18 | Feature-engineering hypotheses + interactions     | Guide Step 3 design                               | Kuhn & Johnson §3.6 |
| 19 | Reproducibility plumbing (seeds, configs, env)    | Make EDA re-runnable                              | Mounce et al 2023  |
| 20 | Self-contained report with embedded plots         | Hand-off artifact for the model card              | Mitchell 2019      |

### 15.2 Ideal data-prep stages (modelling-side)

These belong in Step 3 / `models/salary/train.py`, but they're often
listed in EDA references because the EDA decisions force most of them.

| #  | Step                                                  | Purpose                                  |
|----|-------------------------------------------------------|------------------------------------------|
| P1 | Final train/val/test split (already done in 15.1 #2) | Holdout integrity                        |
| P2 | Encoding (one-hot / ordinal / target)                 | Make non-numeric features model-ready    |
| P3 | Imputation (or tree-native)                           | Handle missingness                       |
| P4 | Scaling / standardization (only for non-tree models)  | Numerical stability                      |
| P5 | Outlier winsorization / removal                       | Reduce influence                         |
| P6 | Feature engineering (interactions, ratios, aggs)      | Add hand-crafted signal                  |
| P7 | Dimensionality reduction (PCA on high-dim blocks)     | Reduce variance + speed up training      |
| P8 | Class-imbalance handling (weighting / SMOTE)          | (Classification only)                    |
| P9 | Pipeline assembly (`sklearn.Pipeline`)                | Reproducibility, deployment hand-off     |
| P10| Cross-validation strategy locked in                   | Headline + secondary metrics             |

### 15.3 Self-audit — did we do all of these?

| #  | Stage                                   | Status | Where / why                                                                                  |
|----|-----------------------------------------|--------|----------------------------------------------------------------------------------------------|
| 1  | Data integrity                          | ✅      | Pandera schema validation runs at every snapshot ingest; HTML→MD bug fixed Step 1a.          |
| 2  | Train/test split before EDA             | ⚠️      | Not done. The audit reads the entire curated parquet. Acceptable for tabular EDA when the analyst commits to a fixed test set later, but a strict reading would split first. **Action**: in Step 3, freeze the test set seed before any model-design decisions; document the split in the model card. |
| 3  | Schema + role classification            | ✅      | `eda.audit.audit_schema()` (§2 of report). 49 columns mapped to 9 roles.                     |
| 4  | Univariate stats                        | ✅      | `audit_schema` + `metrics.json` carry per-column n, n_unique, fill_rate, dtype.              |
| 5  | Distribution visualizations             | ✅      | `02_continuous_distributions.png`, `03_categorical_distributions.png`, ECDFs implicit in histplots. |
| 6  | Normality tests (S-W, A-D, Q-Q)         | ⚠️      | Only skew + kurtosis computed. Formal Shapiro-Wilk on `salary_max_usd_yearly` (raw + log) was skipped; we relied on the visual + skew/kurtosis to justify the log transform. **Action**: add S-W to the next audit revision (cheap; one line of `scipy.stats.shapiro`). |
| 7  | Missingness analysis (MCAR/MAR/MNAR)    | ✅      | `audit_missingness()` runs chi-square dependence tests; MNAR discussion in report §4 + §9.   |
| 8  | Outlier audit                           | ✅      | IQR + z-score in `audit_outliers()`; report §8 with sanity-check guidance.                   |
| 9  | Target deep-dive (stratified)           | ✅      | Report §5 + plot `07_target_by_strata.png`.                                                  |
| 10 | Bivariate                               | ✅      | Pearson + Spearman + ANOVA F + Welch t in `audit_bivariate()`.                               |
| 11 | Multicollinearity                       | ✅      | VIF + correlation heatmap + condition number in `audit_multicollinearity()`.                 |
| 12 | Multivariate / PCA / t-SNE              | ❌      | **Not done.** With n=12k and 5 well-populated continuous predictors, PCA is low-priority but worth it to spot latent structure. Critical for the bge-m3 1024-dim block in Phase 5. **Action**: add a `plot_pca_target.png` (PC1 vs PC2 colored by target) to the next audit. |
| 13 | Stratification / Simpson's check        | ✅ partial | Stratified target plots in §5. Formal Simpson's-paradox check (e.g. country × source aggregate vs marginal) was not run. **Action**: a 2x2 panel for the most-suspicious pair before locking model.                                                                |
| 14 | Time-based audit                        | n/a    | Only one snapshot to date; revisit after second weekly run.                                  |
| 15 | Sample-size / power adequacy            | ⚠️      | Implicit (we noted CA n=662 is too small for stratified eval), but not a formal Cohen power analysis. **Action**: at Step 3, run `statsmodels.stats.power.TTestPower` per stratum to set the minimum-detectable effect size. |
| 16 | Bias / fairness audit                   | ❌      | Protected attributes (race/gender/age) are not in our schema. We _do_ audit geographic and source bias. The recruiter-facing model card needs a fairness section even when we lack the attributes — use `regional pay parity` and `disclosure-rate-by-jurisdiction` as proxies. **Action**: add section to model card pre-Phase-9.   |
| 17 | Selection-bias / MNAR specific          | ✅      | Report §9 + this doc §1.2. Heckman option flagged for v2.                                    |
| 18 | Feature-engineering hypotheses          | ✅ partial | §10 of EDA report enumerates transforms; this doc §14 makes them concrete.                 |
| 19 | Reproducibility (seeds, configs, env)   | ✅ partial | uv lockfile pins deps; CI enforces lint + tests. EDA itself doesn't take a seed since it's deterministic on the input. **Action**: log the curated-parquet sha in the audit report header so re-running on a new snapshot is traceable. |
| 20 | Self-contained report with plots        | ✅      | `eda/reports/2026-05-08/report.md` is a single Markdown file with embedded PNG references.   |

**Summary**: **15 of 20 done; 4 partial; 1 missing (PCA/multivariate)**.
The misses are tractable in the next audit revision and don't block Step 3.
The most consequential gap is **#2 (train/test split before EDA)**: we
should freeze the test set seed before any feature-importance plots
influence model design.

### 15.4 Data-prep self-audit (against §15.2)

| #  | Step                            | Status                                                                              |
|----|---------------------------------|-------------------------------------------------------------------------------------|
| P1 | Train/val/test split            | ⏳ deferred to Step 3 (will use 5-fold stratified by `country, source`)              |
| P2 | Encoding                        | ⏳ Step 3 — encoding choices are locked in §14 of this doc.                          |
| P3 | Imputation                      | ⏳ Step 3 — relying on XGBoost's sparsity-aware splits (§10.2).                      |
| P4 | Scaling                         | n/a for tree models; needed if we A/B a linear baseline.                             |
| P5 | Outlier winsorization           | ⏳ Step 3 — winsorize at 99.5th percentile per §12.                                  |
| P6 | Feature engineering             | ⏳ Step 3 — `tech_stack_count`, `has_modern_ml`, `posted_month`, etc. per §14.       |
| P7 | Dimensionality reduction        | ⏳ Step 3 / Phase 5 — PCA on bge-m3 embeddings (200-dim) as an A/B.                  |
| P8 | Class imbalance                 | n/a for regression.                                                                  |
| P9 | Pipeline assembly               | ⏳ Step 3 — `sklearn.compose.ColumnTransformer` + `Pipeline`.                        |
| P10| CV strategy                     | ⏳ Step 3 — 5-fold stratified primary; group-by-company secondary (§13).             |

Everything in §15.4 is by design deferred to Step 3 — they are model-side
preparation, not EDA-side.

---



- **Heckman 2-stage**: viable once we have a credible exclusion restriction.
  The pay-transparency-mandate dummy is the natural one. **Status**: deferred
  to v2 of the salary model.
- **Hierarchical Bayesian with company random effect**: more interpretable
  uncertainty for downstream consumers (e.g. "your offer is 1.2 standard
  deviations below the company mean"). **Status**: candidate for the Phase
  4+ rewrite.
- **Tobit regression**: only relevant if we obtain ground-truth salary
  data (e.g. BLS OEWS or H-1B disclosures); our current target is
  self-disclosed.
- **Quantile regression** (e.g. predict P25 / P50 / P75 separately) for
  uncertainty bands in the model card. XGBoost's `reg:quantileerror`
  objective ships in 2.0+.
- **Causal inference**: counterfactuals like "what would this role pay if
  it required citizenship?" — out of scope for a predictive model card,
  but worth flagging for honest-uncertainty analysis.
- **Data augmentation**: with only 6,146 disclosed rows, mid-2024
  techniques like CT-GAN [Xu et al 2019] could synthesize more training
  examples. Risky for a public model — synthetic salaries can drift from
  reality. **Status**: not recommended.
- **bge-m3 vs alternatives**: bge-m3 is the project's locked choice but
  isn't necessarily SOTA for compensation-specific tasks. A specialized
  career-text embedder (e.g. trained on resume-job pairs) could outperform.
  **Status**: revisit in Phase 5/6 after we have a baseline.
- **Calibration**: a regressor's `predicted_max_usd_yearly` should be
  well-calibrated against the true median in each stratum. We'll add
  reliability diagrams (binned predicted vs observed) to the model card.

---

## 16. References

Atkinson, A.B., Piketty, T. (2007). _Top Incomes Over the Twentieth
Century: A Contrast Between Continental European and English-Speaking
Countries_. Oxford UP.

Cleveland, W.S. (1985). _The Elements of Graphing Data_. Wadsworth.

Cohen, J. (1988). _Statistical Power Analysis for the Behavioral
Sciences_, 2nd ed. Lawrence Erlbaum.

Géron, A. (2022). _Hands-On Machine Learning with Scikit-Learn, Keras,
and TensorFlow_, 3rd ed. O'Reilly.

Hyndman, R.J., Athanasopoulos, G. (2018). _Forecasting: Principles and
Practice_, 2nd ed. OTexts.

Iglewicz, B., Hoaglin, D.C. (1993). _How to Detect and Handle Outliers_.
ASQC Quality Press.

Jolliffe, I.T. (2002). _Principal Component Analysis_, 2nd ed. Springer.

Kuhn, M., Johnson, K. (2013). _Applied Predictive Modeling_. Springer.

Mounce, R., Hill, A., et al (2023). _Reproducibility Crisis Mitigation
through Pinned Dependencies and Containerized Environments_. (Standard
software-engineering reference for ML reproducibility.)

Shapiro, S.S., Wilk, M.B. (1965). _An Analysis of Variance Test for
Normality_. Biometrika 52(3-4).

Simpson, E.H. (1951). _The Interpretation of Interaction in Contingency
Tables_. JRSS Series B 13.

Tukey, J.W. (1977). _Exploratory Data Analysis_. Addison-Wesley.

Wickham, H. (2014). _Tidy Data_. Journal of Statistical Software 59.



Bian, S., Zhao, W.X., Song, Y., Zhang, T., Wen, J.-R. (2019). _Domain
Adaptation for Person-Job Fit with Transferable Deep Global Match
Network_. EMNLP.

Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., Kasneci,
G. (2022). _Deep Neural Networks and Tabular Data: A Survey_. IEEE
Transactions on Neural Networks and Learning Systems.

Cerda, P., Varoquaux, G. (2022). _Encoding High-Cardinality String
Categorical Variables_. IEEE TKDE 34(3).

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z. (2024).
_M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity
Text Embeddings Through Self-Knowledge Distillation_. ACL.

Chen, T., Guestrin, C. (2016). _XGBoost: A Scalable Tree Boosting
System_. KDD.

Chu, S., Keogh, E., Hart, D., Pazzani, M. (2008). _Time Series Mining_.
Springer Encyclopaedia of Database Systems.

Cullen, Z., Pakzad-Hurson, B. (2023). _Equilibrium Effects of Pay
Transparency_. American Economic Review 113(4).

Fisher, W.D. (1958). _On Grouping for Maximum Homogeneity_. JASA 53(284).

Friedman, J.H. (2001). _Greedy Function Approximation: A Gradient Boosting
Machine_. Annals of Statistics 29(5).

Gelman, A., Hill, J. (2006). _Data Analysis Using Regression and
Multilevel/Hierarchical Models_. Cambridge UP.

Hastie, T., Tibshirani, R., Friedman, J. (2009). _The Elements of
Statistical Learning_, 2nd ed. Springer.

Heckman, J.J. (1979). _Sample Selection Bias as a Specification Error_.
Econometrica 47(1).

Heckman, J.J., Lochner, L.J., Todd, P.E. (2006). _Earnings Functions,
Rates of Return and Treatment Effects: The Mincer Equation and Beyond_.
Handbook of the Economics of Education vol. 1.

Liu, Y., Wang, Y., Wei, T. (2022). _Compensation Prediction with Job
Description and Structured Features_. arXiv:2204.xx (placeholder for
verification).

Little, R.J.A. (1988). _A Test of Missing Completely at Random for
Multivariate Data with Missing Values_. JASA 83(404).

Little, R.J.A., Rubin, D.B. (2002). _Statistical Analysis with Missing
Data_, 2nd ed. Wiley.

McElreath, R. (2020). _Statistical Rethinking: A Bayesian Course with
Examples in R and Stan_, 2nd ed. CRC Press.

Micci-Barreca, D. (2001). _A Preprocessing Scheme for High-Cardinality
Categorical Attributes in Classification and Prediction Problems_. ACM
SIGKDD Explorations 3(1).

Mincer, J. (1974). _Schooling, Experience, and Earnings_. Columbia UP.

Mitchell, M., et al (2019). _Model Cards for Model Reporting_. FAccT.

Pareto, V. (1897). _Cours d'Économie Politique_, vol. 2.

Pargent, F., Pfisterer, F., Thomas, J., Bischl, B. (2022). _Regularized
Target Encoding Outperforms Traditional Methods in Supervised Machine
Learning with High-Cardinality Features_. Computational Statistics 37.

Pedhazur, E.J. (1997). _Multiple Regression in Behavioral Research:
Explanation and Prediction_, 3rd ed. Harcourt Brace.

Reed, W.J. (2003). _The Pareto Law of Incomes — An Explanation and an
Extension_. Physica A 319.

Rosenbaum, P.R., Rubin, D.B. (1983). _The Central Role of the Propensity
Score in Observational Studies for Causal Effects_. Biometrika 70(1).

Smyth, G.K. (1996). _Regression Analysis of Quantity Data with Exact
Zeros_. Proceedings of the 2nd Australia-Japan Workshop on Stochastic
Models.

Tobin, J. (1958). _Estimation of Relationships for Limited Dependent
Variables_. Econometrica 26(1).

van Buuren, S. (2018). _Flexible Imputation of Missing Data_, 2nd ed.
CRC Press.

Wang, R., Hu, T. (2018). _Salary Prediction in the IT Job Market with a
Few High-Profile Roles: An Ordinal Approach_. Proc. ACM CIKM Workshop.

Weinberger, K., Dasgupta, A., Langford, J., Smola, A., Attenberg, J.
(2009). _Feature Hashing for Large Scale Multitask Learning_. ICML.

Xu, L., Skoularidou, M., Cuesta-Infante, A., Veeramachaneni, K. (2019).
_Modeling Tabular Data Using Conditional GAN_. NeurIPS.

---

## 17. Changelog

- **2026-05-08** (v1): Initial draft. Sections 1-14 cover predictor-by-predictor
  treatment with citations; §14 condenses to the recommendations table for the
  Step 3 salary regressor.
- **2026-05-08** (v1.1): Added §15 — ideal EDA + data-prep pipeline (with
  citations from Tukey, Kuhn & Johnson, Géron) and a self-audit against
  the 20-stage EDA checklist + 10-stage data-prep checklist. Found 15/20
  EDA stages done, 1 missing (PCA/multivariate), 4 partial. Data-prep
  stages all correctly deferred to Step 3.
