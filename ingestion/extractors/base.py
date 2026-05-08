"""Shared base class + helpers for ATS extractors.

All extractors are async (httpx.AsyncClient) and use tenacity for retries with
exponential backoff. Per-extractor concurrency is bounded by an asyncio
semaphore so we don't blast a single ATS host. Per-company errors are caught
and recorded in a stats dict — one bad company does not fail the snapshot.
"""

from __future__ import annotations

import asyncio
import hashlib
import html as html_module
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import httpx
from markdownify import markdownify
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ingestion.schema import CanonicalJob, CompanyConfig

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
USER_AGENT = "na-tech-jobs/0.0.1 (+https://github.com/Arjun10g/na-tech-jobs)"


def stable_id(company_slug: str, url: str) -> str:
    """Stable 16-hex-char ID from company slug + canonical URL.

    Matches CLAUDE.md §6: `id = sha256(company_slug + url)[:16]`.
    """
    digest = hashlib.sha256(f"{company_slug}|{url}".encode()).hexdigest()
    return digest[:16]


def hash_payload(payload: Any) -> str:
    """Hash a JSON-serializable payload for change detection."""
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def html_to_markdown(html: str | None) -> str:
    """Convert ATS-provided HTML descriptions to clean markdown.

    Greenhouse double-encodes their content (``&lt;h2&gt;`` rather than
    ``<h2>``); markdownify alone leaves it as escaped text. Unescape first,
    then convert.
    """
    if not html:
        return ""
    decoded = html_module.unescape(html)
    return markdownify(decoded, heading_style="ATX", strip=["script", "style"]).strip()


def make_client() -> httpx.AsyncClient:
    """Construct an httpx.AsyncClient with project defaults."""
    return httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        follow_redirects=True,
    )


class ExtractorError(Exception):
    """Raised when an extractor cannot produce jobs for a single company."""


class Extractor(ABC):
    """Abstract async ATS extractor.

    Subclasses implement `fetch_company`. The base class handles retries,
    concurrency throttling, and per-company error capture.
    """

    source: str = "abstract"
    max_concurrency: int = 4

    def __init__(
        self,
        client: httpx.AsyncClient,
        max_concurrency: int | None = None,
    ) -> None:
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrency or self.max_concurrency)
        self.logger = logging.getLogger(f"ingestion.{self.source}")

    async def get_json(self, url: str, **kwargs: Any) -> Any | None:
        """GET a URL with retries; return parsed JSON, or None for 404."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(
                (
                    httpx.TimeoutException,
                    httpx.RemoteProtocolError,
                    httpx.ConnectError,
                    httpx.ReadError,
                )
            ),
            reraise=True,
        ):
            with attempt:
                async with self.semaphore:
                    response = await self.client.get(url, **kwargs)
                    if response.status_code == 404:
                        return None
                    response.raise_for_status()
                    return response.json()
        return None

    @abstractmethod
    async def fetch_company(self, company: CompanyConfig) -> list[CanonicalJob]:
        """Return canonical jobs for a single company. Must not raise on missing
        boards — return [] and let the caller decide."""

    async def fetch_many(
        self, companies: Iterable[CompanyConfig]
    ) -> tuple[list[CanonicalJob], list[dict[str, Any]]]:
        """Fetch all companies concurrently, capturing per-company errors."""
        targets = [c for c in companies if c.provider == self.source]

        async def _one(c: CompanyConfig) -> tuple[CompanyConfig, list[CanonicalJob], str | None]:
            try:
                jobs = await self.fetch_company(c)
                return c, jobs, None
            except Exception as exc:  # noqa: BLE001 - we want to record any failure
                self.logger.warning("%s/%s failed: %s", self.source, c.slug, exc)
                return c, [], f"{type(exc).__name__}: {exc}"

        results = await asyncio.gather(*[_one(c) for c in targets])

        all_jobs: list[CanonicalJob] = []
        stats: list[dict[str, Any]] = []
        for company, jobs, err in results:
            stats.append(
                {
                    "company": company.slug,
                    "provider": self.source,
                    "jobs": len(jobs),
                    "error": err,
                }
            )
            all_jobs.extend(jobs)
        return all_jobs, stats
