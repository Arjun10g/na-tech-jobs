"""Canonical job schema. Matches CLAUDE.md §6.

The Pydantic model is the source of truth; the Pandera schema validates the
materialised parquet shape after extractors normalize their payloads. Prediction
columns (predicted_salary_usd_v{N}, seniority_label_v{N}, …) are added later by
curated/enrich.py and are not part of the snapshot schema.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

import pandas as pd
import pandera.pandas as pa
from pandera.engines.pandas_engine import DateTime
from pydantic import BaseModel, Field


class RemotePolicy(str, Enum):
    onsite = "onsite"
    hybrid = "hybrid"
    remote = "remote"
    remote_na = "remote-na"


class SalaryPeriod(str, Enum):
    year = "year"
    month = "month"
    day = "day"
    hour = "hour"


class CanonicalJob(BaseModel):
    """Canonical job posting. One row per (company, url) at scrape time."""

    id: str = Field(..., description="sha256(company_slug + url)[:16]; stable across snapshots")
    company_slug: str
    company_name: str
    title: str
    url: str
    location_raw: str | None = None
    country: str | None = Field(None, description="ISO 3166-1 alpha-2: US, CA")
    region: str | None = None
    city: str | None = None
    remote_policy: RemotePolicy | None = None
    seniority_extracted: str | None = None
    role_family_extracted: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = Field(None, description="ISO 4217: USD, CAD")
    salary_period: SalaryPeriod | None = None
    salary_min_usd_yearly: float | None = None
    salary_max_usd_yearly: float | None = None
    salary_disclosed: bool = False
    description_md: str = ""
    posted_at: datetime | None = None
    scraped_at: datetime
    source: str = Field(..., description="extractor name: greenhouse, lever, ashby, …")
    raw_payload_hash: str


CANONICAL_COLUMNS: tuple[str, ...] = tuple(CanonicalJob.model_fields.keys())

REMOTE_POLICY_VALUES: tuple[str, ...] = tuple(p.value for p in RemotePolicy)
SALARY_PERIOD_VALUES: tuple[str, ...] = tuple(p.value for p in SalaryPeriod)
ALLOWED_COUNTRIES: tuple[str, ...] = ("US", "CA")
ALLOWED_CURRENCIES: tuple[str, ...] = ("USD", "CAD")
ALLOWED_SOURCES: tuple[str, ...] = (
    "greenhouse",
    "lever",
    "ashby",
    "workable",
    "smartrecruiters",
    "workday",
)


JOB_SCHEMA = pa.DataFrameSchema(
    columns={
        "id": pa.Column(str, checks=pa.Check.str_length(min_value=16, max_value=16), unique=True),
        "company_slug": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "company_name": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "title": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "url": pa.Column(str, checks=pa.Check.str_startswith("http")),
        "location_raw": pa.Column(str, nullable=True),
        "country": pa.Column(str, nullable=True, checks=pa.Check.isin([*ALLOWED_COUNTRIES, None])),
        "region": pa.Column(str, nullable=True),
        "city": pa.Column(str, nullable=True),
        "remote_policy": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*REMOTE_POLICY_VALUES, None])
        ),
        "seniority_extracted": pa.Column(str, nullable=True),
        "role_family_extracted": pa.Column(str, nullable=True),
        "salary_min": pa.Column(float, nullable=True),
        "salary_max": pa.Column(float, nullable=True),
        "salary_currency": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*ALLOWED_CURRENCIES, None])
        ),
        "salary_period": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*SALARY_PERIOD_VALUES, None])
        ),
        "salary_min_usd_yearly": pa.Column(float, nullable=True),
        "salary_max_usd_yearly": pa.Column(float, nullable=True),
        "salary_disclosed": pa.Column(bool),
        "description_md": pa.Column(str),
        "posted_at": pa.Column(DateTime(tz="UTC"), nullable=True),
        "scraped_at": pa.Column(DateTime(tz="UTC")),
        "source": pa.Column(str, checks=pa.Check.isin(ALLOWED_SOURCES)),
        "raw_payload_hash": pa.Column(str, checks=pa.Check.str_length(min_value=8)),
    },
    strict=False,
    coerce=True,
)


def empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical columns in canonical order."""
    return pd.DataFrame({col: pd.Series(dtype="object") for col in CANONICAL_COLUMNS})


class CompanyConfig(BaseModel):
    """One entry in `ingestion/companies.yaml`."""

    slug: str
    name: str
    provider: str = Field(
        ..., description="greenhouse | lever | ashby | workable | smartrecruiters | workday"
    )
    handle: str = Field(..., description="ATS-specific board handle (subdomain or company key)")
    default_country: str | None = Field(
        None, description="Fallback country when location string is ambiguous (US|CA)."
    )
