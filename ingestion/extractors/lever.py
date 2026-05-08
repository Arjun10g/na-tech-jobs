"""Lever Postings API extractor.

Endpoint: https://api.lever.co/v0/postings/{handle}?mode=json
Docs:     https://help.lever.co/hc/en-us/articles/360037610874

Returns a JSON array of postings with structured `categories`, `salaryRange`,
and HTML descriptions in `description` + `additional`.
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

BASE = "https://api.lever.co/v0/postings"

PERIOD_MAP = {
    "per-year-salary": "year",
    "OneTime": "year",
    "per-hour-salary": "hour",
    "per-month-salary": "month",
    "per-day-salary": "day",
}


def _parse_salary(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "salary_period": None,
    }
    sr = raw.get("salaryRange") or {}
    if not sr:
        return out
    try:
        if sr.get("min") is not None:
            out["salary_min"] = float(sr["min"])
        if sr.get("max") is not None:
            out["salary_max"] = float(sr["max"])
    except (TypeError, ValueError):
        return out
    out["salary_currency"] = sr.get("currency")
    interval = sr.get("interval")
    if interval and interval in PERIOD_MAP:
        out["salary_period"] = PERIOD_MAP[interval]
    elif isinstance(interval, str) and "year" in interval.lower():
        out["salary_period"] = "year"
    return out


def _description(raw: dict[str, Any]) -> str:
    parts: list[str] = []
    desc = raw.get("description") or raw.get("descriptionPlain") or ""
    if desc:
        parts.append(html_to_markdown(desc) if "<" in desc else desc)
    for li in raw.get("lists") or []:
        text = li.get("text") or ""
        content = li.get("content") or ""
        if text:
            parts.append(f"## {text}")
        if content:
            parts.append(html_to_markdown(content) if "<" in content else content)
    extra = raw.get("additional") or raw.get("additionalPlain") or ""
    if extra:
        parts.append(html_to_markdown(extra) if "<" in extra else extra)
    return "\n\n".join(p for p in parts if p).strip()


class LeverExtractor(Extractor):
    source = "lever"

    async def fetch_company(self, company: CompanyConfig) -> list[CanonicalJob]:
        url = f"{BASE}/{company.handle}?mode=json"
        payload = await self.get_json(url)
        if payload is None:
            self.logger.info("lever 404 for %s (handle=%s)", company.slug, company.handle)
            return []
        if not isinstance(payload, list):
            self.logger.warning("lever %s: unexpected payload shape", company.slug)
            return []

        scraped_at = datetime.now(timezone.utc)
        jobs: list[CanonicalJob] = []
        for raw in payload:
            posting_url = raw.get("hostedUrl") or raw.get("applyUrl") or ""
            if not posting_url:
                continue
            title = (raw.get("text") or "").strip()
            if not title:
                continue

            categories = raw.get("categories") or {}
            location_raw = categories.get("location")
            commitment = (categories.get("commitment") or "").lower()
            workplace_type = (raw.get("workplaceType") or "").lower()

            posted_at = None
            if raw.get("createdAt"):
                try:
                    posted_at = datetime.fromtimestamp(
                        int(raw["createdAt"]) / 1000, tz=timezone.utc
                    )
                except (TypeError, ValueError):
                    posted_at = None

            salary = _parse_salary(raw)
            disclosed = salary["salary_min"] is not None or salary["salary_max"] is not None

            description_md = _description(raw)
            if commitment:
                description_md = f"_{commitment}_\n\n{description_md}".strip()
            if workplace_type:
                description_md = f"**Workplace:** {workplace_type}\n\n{description_md}".strip()

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
