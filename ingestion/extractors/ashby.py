"""Ashby Job Board API extractor.

Endpoint: https://api.ashbyhq.com/posting-api/job-board/{handle}?includeCompensation=true
Docs:     https://developers.ashbyhq.com/reference/getjobboardposts

Compensation is structured but optional; we read the simplest representation
(`compensation.compensationTierSummary`) and fall back to the first tier's
components when present.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ingestion.extractors.base import (
    Extractor,
    hash_payload,
    html_to_markdown,
    stable_id,
)
from ingestion.schema import CanonicalJob, CompanyConfig

BASE = "https://api.ashbyhq.com/posting-api/job-board"

INTERVAL_MAP = {
    "1 YEAR": "year",
    "1 MONTH": "month",
    "1 WEEK": "week",
    "1 DAY": "day",
    "1 HOUR": "hour",
}


def _parse_compensation(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "salary_period": None,
    }
    comp = raw.get("compensation") or {}
    tiers = comp.get("compensationTiers") or []
    if not tiers:
        return out
    tier = tiers[0] or {}
    components = tier.get("components") or []
    salary_component = next(
        (c for c in components if (c.get("compensationType") or "").lower() == "salary"),
        components[0] if components else None,
    )
    if not salary_component:
        return out

    value = salary_component.get("compensationValue") or {}
    try:
        if value.get("minValue") is not None:
            out["salary_min"] = float(value["minValue"])
        if value.get("maxValue") is not None:
            out["salary_max"] = float(value["maxValue"])
    except (TypeError, ValueError):
        return out
    out["salary_currency"] = value.get("currencyCode")
    interval = value.get("interval") or salary_component.get("interval")
    if isinstance(interval, str):
        out["salary_period"] = INTERVAL_MAP.get(interval.upper())
        if out["salary_period"] == "week":
            out["salary_period"] = None  # not a canonical period
    return out


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _location_string(raw: dict[str, Any]) -> str | None:
    direct = raw.get("location")
    if direct:
        return direct
    secondary = raw.get("secondaryLocations") or []
    if secondary and isinstance(secondary, list):
        first = secondary[0]
        if isinstance(first, dict):
            return first.get("location")
        if isinstance(first, str):
            return first
    return None


class AshbyExtractor(Extractor):
    source = "ashby"

    async def fetch_company(self, company: CompanyConfig) -> list[CanonicalJob]:
        url = f"{BASE}/{company.handle}?includeCompensation=true"
        payload = await self.get_json(url)
        if payload is None:
            self.logger.info("ashby 404 for %s (handle=%s)", company.slug, company.handle)
            return []

        scraped_at = datetime.now(timezone.utc)
        jobs: list[CanonicalJob] = []
        for raw in payload.get("jobs", []) or []:
            posting_url = raw.get("jobUrl") or raw.get("applyUrl") or ""
            if not posting_url:
                continue
            title = (raw.get("title") or "").strip()
            if not title:
                continue
            description_html = raw.get("descriptionHtml") or ""
            description_plain = raw.get("descriptionPlain") or ""
            description_md = html_to_markdown(description_html) or description_plain

            location_raw = _location_string(raw)
            if raw.get("isRemote"):
                location_raw = f"Remote – {location_raw}" if location_raw else "Remote"

            posted_at = _parse_dt(raw.get("publishedDate") or raw.get("updatedAt"))
            comp = _parse_compensation(raw)
            disclosed = comp["salary_min"] is not None or comp["salary_max"] is not None

            jobs.append(
                CanonicalJob(
                    id=stable_id(company.slug, posting_url),
                    company_slug=company.slug,
                    company_name=company.name,
                    title=title,
                    url=posting_url,
                    location_raw=location_raw,
                    salary_min=comp["salary_min"],
                    salary_max=comp["salary_max"],
                    salary_currency=comp["salary_currency"],
                    salary_period=comp["salary_period"],
                    salary_disclosed=disclosed,
                    description_md=description_md,
                    posted_at=posted_at,
                    scraped_at=scraped_at,
                    source=self.source,
                    raw_payload_hash=hash_payload(raw),
                )
            )
        return jobs
