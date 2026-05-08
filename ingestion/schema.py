"""Canonical job schema. Matches CLAUDE.md §6 + Phase 2 feature-extraction columns.

The Pydantic model is the source of truth; the Pandera schema validates the
materialised parquet shape after extractors normalize their payloads. Prediction
columns (predicted_salary_usd_v{N}, seniority_label_v{N}, …) are added later by
curated/enrich.py and are not part of the snapshot schema.

Phase 2 introduces ~22 nullable feature columns (salary mining, requirements,
compensation extras, contract type, posting quality, language requirements, …)
populated by ``ingestion/feature_extraction/`` via a regex-first cascade with
NuExtract LLM fallback. ``extraction_meta`` carries per-field provenance.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

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


class Education(str, Enum):
    high_school = "high_school"
    associates = "associates"
    bachelors = "bachelors"
    masters = "masters"
    phd = "phd"


class ClearanceLevel(str, Enum):
    public_trust = "public_trust"
    confidential = "confidential"
    secret = "secret"
    top_secret = "top_secret"
    ts_sci = "ts_sci"


class SponsorshipPolicy(str, Enum):
    yes = "yes"
    no = "no"
    unspecified = "unspecified"


class EquityForm(str, Enum):
    rsu = "rsu"
    options = "options"
    profit_sharing = "profit_sharing"
    other = "other"


class BonusType(str, Enum):
    signing = "signing"
    annual = "annual"
    performance = "performance"
    retention = "retention"


class ContractType(str, Enum):
    full_time = "full_time"
    part_time = "part_time"
    contract = "contract"
    internship = "internship"
    temporary = "temporary"


class ManagerRole(str, Enum):
    ic = "ic"
    tech_lead = "tech_lead"
    manager = "manager"
    senior_manager = "senior_manager"
    director = "director"
    exec = "exec"


class PostingQuality(str, Enum):
    real = "real"
    evergreen_pool = "evergreen_pool"
    talent_community = "talent_community"
    reposted = "reposted"


EDUCATION_VALUES = tuple(e.value for e in Education)
CLEARANCE_VALUES = tuple(e.value for e in ClearanceLevel)
SPONSORSHIP_VALUES = tuple(e.value for e in SponsorshipPolicy)
EQUITY_FORM_VALUES = tuple(e.value for e in EquityForm)
BONUS_TYPE_VALUES = tuple(e.value for e in BonusType)
CONTRACT_TYPE_VALUES = tuple(e.value for e in ContractType)
MANAGER_ROLE_VALUES = tuple(e.value for e in ManagerRole)
POSTING_QUALITY_VALUES = tuple(e.value for e in PostingQuality)


class CanonicalJob(BaseModel):
    """Canonical job posting. One row per (company, url) at scrape time."""

    # ── Identity ────────────────────────────────────────────────────────────
    id: str = Field(..., description="sha256(company_slug + url)[:16]; stable across snapshots")
    company_slug: str
    company_name: str
    title: str
    url: str

    # ── Location ────────────────────────────────────────────────────────────
    location_raw: str | None = None
    country: str | None = Field(None, description="ISO 3166-1 alpha-2: US, CA")
    region: str | None = None
    city: str | None = None
    remote_policy: RemotePolicy | None = None

    # ── Title-derived (replaced by ML classifiers in Phase 4) ───────────────
    seniority_extracted: str | None = None
    role_family_extracted: str | None = None

    # ── Structured salary (from ATS structured fields) ──────────────────────
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = Field(None, description="ISO 4217: USD, CAD")
    salary_period: SalaryPeriod | None = None
    salary_min_usd_yearly: float | None = None
    salary_max_usd_yearly: float | None = None
    salary_disclosed: bool = False

    # ── Phase 2 extracted features (regex-first cascade, LLM fallback) ──────
    min_years_experience: int | None = None
    max_years_experience: int | None = None
    min_education: Education | None = None
    requires_security_clearance: bool | None = None
    clearance_level: ClearanceLevel | None = None
    requires_citizenship: list[str] | None = None
    offers_visa_sponsorship: SponsorshipPolicy | None = None
    offers_relocation: bool | None = None
    offers_equity: bool | None = None
    equity_form: EquityForm | None = None
    bonus_mentioned: bool | None = None
    bonus_type: BonusType | None = None
    max_travel_percent: int | None = None
    contract_type: ContractType | None = None
    on_call_required: bool | None = None
    manager_role: ManagerRole | None = None
    direct_reports_count: int | None = None
    posting_quality: PostingQuality | None = None
    language_requirements: list[str] | None = None
    tech_stack: list[str] | None = None
    industry_experience: list[str] | None = None
    team_or_department: str | None = None

    # Per-field provenance: {feature_name: {"source": "regex|llm|structured",
    # "confidence": 0.0-1.0, "rule_id": str | None}}
    extraction_meta: dict[str, Any] | None = None
    extraction_version: str = "v1"

    # ── Content + provenance ────────────────────────────────────────────────
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
        # Phase 2 extracted features. Lists/dicts roundtrip through parquet as
        # objects, which Pandera permits via dtype=object + nullable.
        "min_years_experience": pa.Column("Int64", nullable=True),
        "max_years_experience": pa.Column("Int64", nullable=True),
        "min_education": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*EDUCATION_VALUES, None])
        ),
        "requires_security_clearance": pa.Column("boolean", nullable=True),
        "clearance_level": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*CLEARANCE_VALUES, None])
        ),
        "requires_citizenship": pa.Column(object, nullable=True),
        "offers_visa_sponsorship": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*SPONSORSHIP_VALUES, None])
        ),
        "offers_relocation": pa.Column("boolean", nullable=True),
        "offers_equity": pa.Column("boolean", nullable=True),
        "equity_form": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*EQUITY_FORM_VALUES, None])
        ),
        "bonus_mentioned": pa.Column("boolean", nullable=True),
        "bonus_type": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*BONUS_TYPE_VALUES, None])
        ),
        "max_travel_percent": pa.Column("Int64", nullable=True),
        "contract_type": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*CONTRACT_TYPE_VALUES, None])
        ),
        "on_call_required": pa.Column("boolean", nullable=True),
        "manager_role": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*MANAGER_ROLE_VALUES, None])
        ),
        "direct_reports_count": pa.Column("Int64", nullable=True),
        "posting_quality": pa.Column(
            str, nullable=True, checks=pa.Check.isin([*POSTING_QUALITY_VALUES, None])
        ),
        "language_requirements": pa.Column(object, nullable=True),
        "tech_stack": pa.Column(object, nullable=True),
        "industry_experience": pa.Column(object, nullable=True),
        "team_or_department": pa.Column(str, nullable=True),
        "extraction_meta": pa.Column(object, nullable=True),
        "extraction_version": pa.Column(str),
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
