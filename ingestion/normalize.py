"""Post-extraction normalization.

Each extractor returns a `CanonicalJob` with raw location, salary, and
description fields. This module derives:
- country / region / city from `location_raw`
- remote_policy from location + description hints
- salary_{min,max}_usd_yearly via FX + period conversion
- title cleaning (trims location suffixes that some ATS append)
- naive seniority + role-family extraction from the title (replaced by ML
  classifiers in Phase 4)

It also flags Quebec French postings for filtering. CLAUDE.md §3 locks v1
to English-only; the heuristic here is intentionally conservative — false
positives are preferred over leaking French rows into the dataset.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ingestion.schema import CanonicalJob, RemotePolicy, SalaryPeriod

# --- FX + period conversion (v1 hardcode; CLAUDE.md §11 risk noted) ----------

FX_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "CAD": 0.73,
}

PERIOD_TO_YEARLY: dict[str, float] = {
    SalaryPeriod.year.value: 1.0,
    SalaryPeriod.month.value: 12.0,
    SalaryPeriod.day.value: 260.0,  # ~5 days/wk × 52 wks
    SalaryPeriod.hour.value: 2080.0,  # 40 hrs/wk × 52 wks
}

# --- US states + Canadian provinces ------------------------------------------

US_STATES: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}

CA_PROVINCES: dict[str, str] = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "MB": "Manitoba",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
    "SK": "Saskatchewan",
    "YT": "Yukon",
}

US_NAME_TO_ABBR = {v.lower(): k for k, v in US_STATES.items()}
CA_NAME_TO_ABBR = {v.lower(): k for k, v in CA_PROVINCES.items()}
# Common aliases / accented forms
CA_NAME_TO_ABBR["québec"] = "QC"
CA_NAME_TO_ABBR["montréal"] = "QC"

US_KEYWORDS = re.compile(r"\b(united states|usa|u\.s\.a\.|u\.s\.)\b", re.IGNORECASE)
CA_KEYWORDS = re.compile(r"\bcanada\b", re.IGNORECASE)
REMOTE_RE = re.compile(r"\bremote\b", re.IGNORECASE)
HYBRID_RE = re.compile(r"\bhybrid\b", re.IGNORECASE)
ONSITE_RE = re.compile(r"\bon[- ]?site\b|\bin[- ]?office\b", re.IGNORECASE)

# --- Quebec French filter ----------------------------------------------------

# Naive but conservative: titles starting with French job-words almost always
# indicate a French posting. Tested against several Canadian ATS handles.
FRENCH_TITLE_RE = re.compile(
    r"^\s*(ingénieur(e)?|développeur(euse)?|programmeur(euse)?|analyste|"
    r"spécialiste|conseiller(ère)?|stagiaire|adjoint(e)?|chef|responsable|"
    r"technicien(ne)?|gestionnaire|directeur(rice)?|architecte|scientifique des données)\b",
    re.IGNORECASE,
)
# Description-level marker: lots of French stop-words in the first 600 chars.
FRENCH_DESC_HITS = re.compile(
    r"\b(nous|avec|notre|leur|cette|votre|pour|dans|qui|sont|aux?|des|"
    r"été|être|équipe|développ|expérience|exigences|responsabilités)\b",
    re.IGNORECASE,
)

# --- Title-derived signals (cheap, replaced by ML in phase 4) ----------------

SENIORITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("intern", re.compile(r"\b(intern|co-?op|stagiaire)\b", re.IGNORECASE)),
    (
        "junior",
        re.compile(
            r"\b(junior|jr\.?|associate|entry[- ]level|new grad|graduate|grad)\b", re.IGNORECASE
        ),
    ),
    ("staff", re.compile(r"\b(staff)\b", re.IGNORECASE)),
    ("principal", re.compile(r"\b(principal|distinguished|fellow)\b", re.IGNORECASE)),
    ("director", re.compile(r"\b(director|head of|vp|vice[- ]president)\b", re.IGNORECASE)),
    ("manager", re.compile(r"\b(manager|mgr\.?|lead)\b", re.IGNORECASE)),
    ("senior", re.compile(r"\b(senior|sr\.?|sénior)\b", re.IGNORECASE)),
]

ROLE_FAMILY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "MLE",
        re.compile(
            r"\b(machine learning engineer|ml engineer|mle\b|deep learning engineer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "RS",
        re.compile(r"\b(research scientist|research engineer|applied scientist)\b", re.IGNORECASE),
    ),
    ("DS", re.compile(r"\b(data scientist|ml scientist|quant(itative)?)\b", re.IGNORECASE)),
    ("DE", re.compile(r"\b(data engineer|analytics engineer|etl engineer)\b", re.IGNORECASE)),
    (
        "DA",
        re.compile(
            r"\b(data analyst|business analyst|product analyst|bi analyst)\b", re.IGNORECASE
        ),
    ),
    (
        "AS",
        re.compile(r"\b(analytics scientist|growth scientist|decision scientist)\b", re.IGNORECASE),
    ),
    (
        "SWE-ML",
        re.compile(
            r"\b(ai engineer|ml platform|llm engineer|gen[- ]?ai engineer)\b", re.IGNORECASE
        ),
    ),
    ("Manager", re.compile(r"\b(manager|director|head of|vp of)\b", re.IGNORECASE)),
]

LOCATION_SUFFIX_RE = re.compile(r"\s*[\(\[][^)\]]+[\)\]]\s*$")


def _strip_title_suffix(title: str) -> str:
    cleaned = LOCATION_SUFFIX_RE.sub("", title).strip()
    return cleaned or title


def extract_seniority(title: str) -> str | None:
    for label, pattern in SENIORITY_PATTERNS:
        if pattern.search(title):
            return label
    return "mid"  # most permissive default; ML classifier replaces this in Phase 4


def extract_role_family(title: str) -> str | None:
    for label, pattern in ROLE_FAMILY_PATTERNS:
        if pattern.search(title):
            return label
    return "Other"


# --- Location parsing --------------------------------------------------------


def _is_us_state_token(token: str) -> str | None:
    t = token.strip()
    if t.upper() in US_STATES:
        return t.upper()
    if t.lower() in US_NAME_TO_ABBR:
        return US_NAME_TO_ABBR[t.lower()]
    return None


def _is_ca_province_token(token: str) -> str | None:
    t = token.strip()
    if t.upper() in CA_PROVINCES:
        return t.upper()
    if t.lower() in CA_NAME_TO_ABBR:
        return CA_NAME_TO_ABBR[t.lower()]
    return None


def parse_location(
    location_raw: str | None,
    default_country: str | None = None,
) -> dict[str, Any]:
    """Best-effort parse of an ATS location string.

    Returns dict with keys: country, region, city, remote_policy. All optional.
    """
    out: dict[str, Any] = {
        "country": None,
        "region": None,
        "city": None,
        "remote_policy": None,
    }
    if not location_raw:
        out["country"] = default_country
        return out

    s = location_raw.strip()

    # Remote / hybrid / onsite signal
    if REMOTE_RE.search(s):
        out["remote_policy"] = RemotePolicy.remote.value
    elif HYBRID_RE.search(s):
        out["remote_policy"] = RemotePolicy.hybrid.value
    elif ONSITE_RE.search(s):
        out["remote_policy"] = RemotePolicy.onsite.value

    # Country hints
    if CA_KEYWORDS.search(s):
        out["country"] = "CA"
    elif US_KEYWORDS.search(s):
        out["country"] = "US"

    # Tokenize on commas, slashes, pipes, and " - " / " – " / " — " separators
    parts = [p.strip(" -–—") for p in re.split(r"[,/|]|\s+[-–—]\s+", s) if p.strip()]

    # Standalone country abbreviations (US, USA) — must come before state detection
    # since "US" isn't a state code but is a common ATS country marker.
    for token in parts:
        upper = token.strip().rstrip(".").upper().replace(".", "")
        if upper in {"US", "USA"}:
            out["country"] = "US"
            break
        if upper == "CANADA":
            out["country"] = "CA"
            break

    for token in parts:
        prov = _is_ca_province_token(token)
        if prov:
            out["country"] = "CA"
            out["region"] = prov
            break
        state = _is_us_state_token(token)
        if state:
            out["country"] = "US"
            out["region"] = state
            break

    # City: first non-region, non-country, non-remote token
    region_set = {out["region"]} if out["region"] else set()
    for token in parts:
        if not token:
            continue
        upper = token.strip().rstrip(".").upper().replace(".", "")
        if upper in {"US", "USA", "CANADA"}:
            continue
        if REMOTE_RE.search(token) or HYBRID_RE.search(token) or ONSITE_RE.search(token):
            continue
        if CA_KEYWORDS.search(token) or US_KEYWORDS.search(token):
            continue
        if _is_us_state_token(token) or _is_ca_province_token(token):
            continue
        if token.upper() in region_set:
            continue
        out["city"] = token.strip()
        break

    # If we still have no country, fall back to the company default
    if out["country"] is None:
        out["country"] = default_country

    # If remote with a region we can keep, treat as remote-na when in NA
    if out["remote_policy"] == RemotePolicy.remote.value and out["country"] in {"US", "CA"}:
        out["remote_policy"] = RemotePolicy.remote_na.value

    return out


# --- Salary normalization ----------------------------------------------------


def normalize_salary(
    salary_min: float | None,
    salary_max: float | None,
    currency: str | None,
    period: str | None,
) -> tuple[float | None, float | None]:
    """Convert salary min/max to USD/year. Return (None, None) on bad inputs."""
    if salary_min is None and salary_max is None:
        return None, None
    if currency not in FX_TO_USD:
        return None, None
    fx = FX_TO_USD[currency]
    period_factor = PERIOD_TO_YEARLY.get(period or SalaryPeriod.year.value)
    if period_factor is None:
        return None, None

    def conv(v: float | None) -> float | None:
        if v is None:
            return None
        # Heuristic: numbers under 1000 are clearly hourly; some ATS report annual
        # but mark interval=hour incorrectly. Trust the period field.
        return float(v) * fx * period_factor

    return conv(salary_min), conv(salary_max)


# --- French detection --------------------------------------------------------


def is_likely_french(job: CanonicalJob) -> bool:
    """Conservative French-content detector. Flags Quebec French postings."""
    if FRENCH_TITLE_RE.search(job.title):
        return True
    head = (job.description_md or "")[:800]
    hits = len(FRENCH_DESC_HITS.findall(head))
    in_quebec = job.region == "QC" or "québec" in (job.location_raw or "").lower()
    return hits >= 5 and in_quebec


# --- Top-level pipeline ------------------------------------------------------


def normalize(job: CanonicalJob, default_country: str | None = None) -> CanonicalJob:
    """Apply all derivations; returns a new CanonicalJob (does not mutate)."""
    title = _strip_title_suffix(job.title)

    loc = parse_location(job.location_raw, default_country)

    salary_min_usd, salary_max_usd = normalize_salary(
        job.salary_min, job.salary_max, job.salary_currency, job.salary_period
    )

    # model_copy(update=...) skips validation — convert raw strings to enum
    # instances explicitly so model_dump() doesn't emit serializer warnings.
    remote_policy = RemotePolicy(loc["remote_policy"]) if loc["remote_policy"] else None

    return job.model_copy(
        update={
            "title": title,
            "country": loc["country"],
            "region": loc["region"],
            "city": loc["city"],
            "remote_policy": remote_policy,
            "seniority_extracted": extract_seniority(title),
            "role_family_extracted": extract_role_family(title),
            "salary_min_usd_yearly": salary_min_usd,
            "salary_max_usd_yearly": salary_max_usd,
        }
    )


# --- Time helpers ------------------------------------------------------------


def utc_now_iso() -> str:
    """Helper for the orchestrator to stamp snapshot dirs (`YYYY-MM-DD`)."""
    return datetime.utcnow().strftime("%Y-%m-%d")
