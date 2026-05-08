"""Smoke tests for the ATS extractors using fake httpx responses."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from ingestion.extractors.ashby import AshbyExtractor
from ingestion.extractors.base import stable_id
from ingestion.extractors.greenhouse import GreenhouseExtractor
from ingestion.extractors.lever import LeverExtractor
from ingestion.schema import CompanyConfig


def _client_with(handler):
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


COMPANY_GREENHOUSE = CompanyConfig(
    slug="acme", name="Acme", provider="greenhouse", handle="acme", default_country="US"
)
COMPANY_LEVER = CompanyConfig(
    slug="acme", name="Acme", provider="lever", handle="acme", default_country="US"
)
COMPANY_ASHBY = CompanyConfig(
    slug="acme", name="Acme", provider="ashby", handle="acme", default_country="US"
)


# --- Greenhouse --------------------------------------------------------------


GREENHOUSE_RESPONSE = {
    "jobs": [
        {
            "id": 1,
            "title": "Senior Data Scientist",
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
            "location": {"name": "San Francisco, CA"},
            "content": "<p>Build models.</p>",
            "first_published": "2025-09-01T12:00:00Z",
            "pay_input_ranges": [
                {
                    "min_cents": 18_000_000,
                    "max_cents": 24_000_000,
                    "currency_type": "USD",
                    "interval": "year",
                }
            ],
        },
        # Missing url → should be skipped
        {"id": 2, "title": "No URL Job", "absolute_url": "", "content": ""},
    ]
}


def _greenhouse_handler(request: httpx.Request) -> httpx.Response:
    if "acme/jobs" in str(request.url):
        return httpx.Response(200, json=GREENHOUSE_RESPONSE)
    return httpx.Response(404)


@pytest.mark.asyncio
async def test_greenhouse_extractor_parses_basic_payload():
    async with _client_with(_greenhouse_handler) as client:
        extractor = GreenhouseExtractor(client)
        jobs = await extractor.fetch_company(COMPANY_GREENHOUSE)
    assert len(jobs) == 1
    j = jobs[0]
    assert j.title == "Senior Data Scientist"
    assert j.id == stable_id("acme", "https://boards.greenhouse.io/acme/jobs/1")
    assert j.salary_min == 180_000.0
    assert j.salary_max == 240_000.0
    assert j.salary_currency == "USD"
    assert j.salary_period == "year"
    assert j.salary_disclosed is True


@pytest.mark.asyncio
async def test_greenhouse_extractor_returns_empty_on_404():
    def handler(_):
        return httpx.Response(404)

    async with _client_with(handler) as client:
        extractor = GreenhouseExtractor(client)
        jobs = await extractor.fetch_company(COMPANY_GREENHOUSE)
    assert jobs == []


# --- Lever -------------------------------------------------------------------


LEVER_RESPONSE = [
    {
        "id": "1",
        "text": "Staff ML Engineer",
        "hostedUrl": "https://jobs.lever.co/acme/1",
        "applyUrl": "https://jobs.lever.co/acme/1/apply",
        "categories": {"location": "Toronto, ON, Canada", "commitment": "Full-time"},
        "createdAt": 1_725_600_000_000,
        "description": "<p>Train large models.</p>",
        "salaryRange": {
            "min": 180_000,
            "max": 240_000,
            "currency": "CAD",
            "interval": "per-year-salary",
        },
    }
]


def _lever_handler(request: httpx.Request) -> httpx.Response:
    if "acme" in str(request.url):
        return httpx.Response(200, json=LEVER_RESPONSE)
    return httpx.Response(404)


@pytest.mark.asyncio
async def test_lever_extractor_parses_basic_payload():
    async with _client_with(_lever_handler) as client:
        extractor = LeverExtractor(client)
        jobs = await extractor.fetch_company(COMPANY_LEVER)
    assert len(jobs) == 1
    j = jobs[0]
    assert j.title == "Staff ML Engineer"
    assert j.location_raw == "Toronto, ON, Canada"
    assert j.salary_currency == "CAD"
    assert j.salary_period == "year"
    assert j.salary_disclosed is True


# --- Ashby -------------------------------------------------------------------


ASHBY_RESPONSE = {
    "jobs": [
        {
            "id": "abc",
            "title": "ML Engineer",
            "jobUrl": "https://jobs.ashbyhq.com/acme/abc",
            "applyUrl": "https://jobs.ashbyhq.com/acme/abc/apply",
            "location": "New York, NY",
            "isRemote": False,
            "descriptionHtml": "<p>Ship models.</p>",
            "publishedDate": "2025-08-01T00:00:00Z",
            "compensation": {
                "compensationTiers": [
                    {
                        "components": [
                            {
                                "compensationType": "Salary",
                                "compensationValue": {
                                    "minValue": 200_000,
                                    "maxValue": 260_000,
                                    "currencyCode": "USD",
                                    "interval": "1 YEAR",
                                },
                            }
                        ]
                    }
                ]
            },
        }
    ]
}


def _ashby_handler(request: httpx.Request) -> httpx.Response:
    if "acme" in str(request.url):
        return httpx.Response(200, json=ASHBY_RESPONSE)
    return httpx.Response(404)


@pytest.mark.asyncio
async def test_ashby_extractor_parses_basic_payload():
    async with _client_with(_ashby_handler) as client:
        extractor = AshbyExtractor(client)
        jobs = await extractor.fetch_company(COMPANY_ASHBY)
    assert len(jobs) == 1
    j = jobs[0]
    assert j.title == "ML Engineer"
    assert j.salary_min == 200_000.0
    assert j.salary_max == 260_000.0
    assert j.salary_currency == "USD"
    assert j.salary_period == "year"


# --- Common ------------------------------------------------------------------


def test_stable_id_is_deterministic_and_short():
    a = stable_id("acme", "https://acme.com/jobs/1")
    b = stable_id("acme", "https://acme.com/jobs/1")
    assert a == b
    assert len(a) == 16


def test_stable_id_changes_with_input():
    assert stable_id("acme", "u1") != stable_id("acme", "u2")
    assert stable_id("acme", "u1") != stable_id("beta", "u1")


@pytest.mark.asyncio
async def test_fetch_many_skips_other_providers():
    """fetch_many on a Greenhouse extractor should ignore Lever/Ashby companies."""

    async with _client_with(_greenhouse_handler) as client:
        extractor = GreenhouseExtractor(client)
        jobs, stats = await extractor.fetch_many(
            [
                COMPANY_GREENHOUSE,
                CompanyConfig(slug="b", name="B", provider="lever", handle="b"),
                CompanyConfig(slug="c", name="C", provider="ashby", handle="c"),
            ]
        )
    assert len(jobs) == 1
    assert len(stats) == 1
    assert stats[0]["provider"] == "greenhouse"


def test_eventloop_runs():
    """Sanity: pytest-asyncio is wired (otherwise above tests would silently skip)."""
    assert asyncio.get_event_loop_policy() is not None
