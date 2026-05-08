"""Greenhouse Job Boards API extractor.

Endpoint: https://boards-api.greenhouse.io/v1/boards/{handle}/jobs?content=true
Docs:     https://developers.greenhouse.io/job-board.html

The `content=true` flag returns the rendered HTML description inline so we
don't have to fetch each posting individually.
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

BASE = "https://boards-api.greenhouse.io/v1/boards"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_salary(job: dict[str, Any]) -> dict[str, Any]:
    """Greenhouse exposes pay ranges via `pay_input_ranges` on some boards.

    Schema is undocumented + inconsistent — handle the common shape and bail
    quietly otherwise.
    """
    out: dict[str, Any] = {
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "salary_period": None,
    }
    ranges = job.get("pay_input_ranges") or []
    if not ranges or not isinstance(ranges, list):
        return out
    first = ranges[0] or {}
    try:
        if first.get("min_cents") is not None:
            out["salary_min"] = float(first["min_cents"]) / 100
        if first.get("max_cents") is not None:
            out["salary_max"] = float(first["max_cents"]) / 100
    except (TypeError, ValueError):
        return out
    out["salary_currency"] = first.get("currency_type")
    interval = (first.get("interval") or "").lower()
    if "year" in interval or "annual" in interval:
        out["salary_period"] = "year"
    elif "hour" in interval:
        out["salary_period"] = "hour"
    elif "month" in interval:
        out["salary_period"] = "month"
    elif "day" in interval:
        out["salary_period"] = "day"
    return out


class GreenhouseExtractor(Extractor):
    source = "greenhouse"

    async def fetch_company(self, company: CompanyConfig) -> list[CanonicalJob]:
        url = f"{BASE}/{company.handle}/jobs?content=true"
        payload = await self.get_json(url)
        if payload is None:
            self.logger.info("greenhouse 404 for %s (handle=%s)", company.slug, company.handle)
            return []

        scraped_at = datetime.now(timezone.utc)
        jobs: list[CanonicalJob] = []
        for raw in payload.get("jobs", []) or []:
            posting_url = raw.get("absolute_url") or ""
            if not posting_url:
                continue
            title = (raw.get("title") or "").strip()
            if not title:
                continue
            location_raw = (raw.get("location") or {}).get("name")
            description_md = html_to_markdown(raw.get("content"))
            posted_at = _parse_dt(raw.get("first_published") or raw.get("updated_at"))
            salary = _parse_salary(raw)
            disclosed = salary["salary_min"] is not None or salary["salary_max"] is not None

            jobs.append(
                CanonicalJob(
                    id=stable_id(company.slug, posting_url),
                    company_slug=company.slug,
                    company_name=company.name,
                    title=title,
                    url=posting_url,
                    location_raw=location_raw,
                    salary_min=salary["salary_min"],
                    salary_max=salary["salary_max"],
                    salary_currency=salary["salary_currency"],
                    salary_period=salary["salary_period"],
                    salary_disclosed=disclosed,
                    description_md=description_md,
                    posted_at=posted_at,
                    scraped_at=scraped_at,
                    source=self.source,
                    raw_payload_hash=hash_payload(raw),
                )
            )
        return jobs
