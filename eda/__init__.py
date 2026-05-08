"""Statistical / exploratory audit run before modelling.

The audit reads ``data/curated/jobs.parquet``, classifies columns by data type
and role (predictor vs metadata vs target), runs the standard pre-modelling
checks (distributions, missingness, multicollinearity, outliers, bivariate
relationships), and writes a self-contained ``data/eda/report.md`` with
embedded visuals plus a machine-readable ``metrics.json``.
"""

from eda.audit import run_audit

__all__ = ["run_audit"]
