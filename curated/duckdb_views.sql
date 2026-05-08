-- DuckDB views over the curated layer.
--
-- Usage (one-shot session):
--     duckdb -c ".read curated/duckdb_views.sql; SELECT * FROM v_disclosed LIMIT 5"
-- Or from Python:
--     con = duckdb.connect()
--     con.execute(open('curated/duckdb_views.sql').read())
--     con.execute("SELECT * FROM v_by_country").df()

-- ── Active jobs (latest snapshot, deduplicated) ───────────────────────────────
CREATE OR REPLACE VIEW v_active_jobs AS
SELECT * FROM read_parquet('data/curated/jobs.parquet');

-- ── Disclosed-salary subset: training population for the salary regressor ────
CREATE OR REPLACE VIEW v_disclosed AS
SELECT *
FROM v_active_jobs
WHERE salary_disclosed = TRUE
  AND salary_max_usd_yearly IS NOT NULL;

-- ── Aggregations by country (US/CA) ──────────────────────────────────────────
CREATE OR REPLACE VIEW v_by_country AS
SELECT
    country,
    COUNT(*)                                       AS n_jobs,
    SUM(CASE WHEN salary_disclosed THEN 1 END)     AS n_disclosed,
    AVG(salary_max_usd_yearly)                     AS avg_max_usd,
    APPROX_QUANTILE(salary_max_usd_yearly, 0.5)    AS median_max_usd,
    APPROX_QUANTILE(salary_max_usd_yearly, 0.75)   AS p75_max_usd
FROM v_active_jobs
GROUP BY country
ORDER BY n_jobs DESC;

-- ── Aggregations by ATS source ───────────────────────────────────────────────
CREATE OR REPLACE VIEW v_by_source AS
SELECT
    source,
    COUNT(*)                                       AS n_jobs,
    AVG(CASE WHEN salary_disclosed THEN 1.0 ELSE 0.0 END) AS disclosure_rate,
    AVG(salary_max_usd_yearly)                     AS avg_max_usd
FROM v_active_jobs
GROUP BY source
ORDER BY n_jobs DESC;

-- ── Aggregations by role family (DS / MLE / DE / DA / RS / SWE-ML / Manager) ─
CREATE OR REPLACE VIEW v_by_role_family AS
SELECT
    role_family_extracted                          AS role_family,
    COUNT(*)                                       AS n_jobs,
    AVG(CASE WHEN salary_disclosed THEN 1.0 ELSE 0.0 END) AS disclosure_rate,
    APPROX_QUANTILE(salary_max_usd_yearly, 0.5)    AS median_max_usd,
    AVG(min_years_experience)                      AS avg_min_yoe
FROM v_active_jobs
GROUP BY role_family_extracted
ORDER BY n_jobs DESC;

-- ── Aggregations by seniority ────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_by_seniority AS
SELECT
    seniority_extracted                            AS seniority,
    COUNT(*)                                       AS n_jobs,
    APPROX_QUANTILE(salary_max_usd_yearly, 0.5)    AS median_max_usd,
    APPROX_QUANTILE(salary_max_usd_yearly, 0.25)   AS p25_max_usd,
    APPROX_QUANTILE(salary_max_usd_yearly, 0.75)   AS p75_max_usd
FROM v_active_jobs
WHERE salary_disclosed = TRUE
GROUP BY seniority_extracted;

-- ── Per-company yield: how many jobs each board contributes ──────────────────
CREATE OR REPLACE VIEW v_by_company AS
SELECT
    company_slug,
    company_name,
    source,
    COUNT(*)                                       AS n_jobs,
    SUM(CASE WHEN salary_disclosed THEN 1 END)     AS n_disclosed
FROM v_active_jobs
GROUP BY company_slug, company_name, source
ORDER BY n_jobs DESC;

-- ── Remote-policy mix ────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_remote_mix AS
SELECT
    COALESCE(remote_policy, '<unknown>')           AS remote_policy,
    COUNT(*)                                       AS n_jobs,
    AVG(salary_max_usd_yearly)                     AS avg_max_usd
FROM v_active_jobs
GROUP BY 1
ORDER BY n_jobs DESC;
