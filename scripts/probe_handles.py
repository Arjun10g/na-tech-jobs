"""Probe candidate ATS handles in parallel and report which are live.

Run:
    uv run python -m scripts.probe_handles            # full sweep, default candidates
    uv run python -m scripts.probe_handles -o data/probe.yaml

The script tries multiple (provider, handle) guesses per company and keeps
the first one that returns ≥1 job. Output is a YAML block ready to paste
into ``ingestion/companies.yaml``.

Used during Phase 1.5 expansion (see MAINTENANCE.md). Not on the regular
runtime path — keep here for the next time we want to grow the registry.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx

GREENHOUSE = "https://boards-api.greenhouse.io/v1/boards/{handle}/jobs?content=false"
LEVER = "https://api.lever.co/v0/postings/{handle}?mode=json"
ASHBY = "https://api.ashbyhq.com/posting-api/job-board/{handle}"


@dataclass
class Candidate:
    slug: str
    name: str
    default_country: str
    options: list[tuple[str, str]]  # ordered (provider, handle) guesses


CANDIDATES: list[Candidate] = [
    # ── Existing seeds we keep ────────────────────────────────────────────
    Candidate("stripe", "Stripe", "US", [("greenhouse", "stripe")]),
    Candidate("airbnb", "Airbnb", "US", [("greenhouse", "airbnb")]),
    Candidate("instacart", "Instacart", "US", [("greenhouse", "instacart")]),
    Candidate("robinhood", "Robinhood", "US", [("greenhouse", "robinhood")]),
    Candidate("brex", "Brex", "US", [("greenhouse", "brex")]),
    Candidate("ramp", "Ramp", "US", [("greenhouse", "ramp")]),
    Candidate("datadog", "Datadog", "US", [("greenhouse", "datadog")]),
    Candidate("mongodb", "MongoDB", "US", [("greenhouse", "mongodb")]),
    Candidate("snowflake", "Snowflake", "US", [("greenhouse", "snowflake")]),
    Candidate("twilio", "Twilio", "US", [("greenhouse", "twilio")]),
    Candidate("anthropic", "Anthropic", "US", [("greenhouse", "anthropic")]),
    Candidate("openai", "OpenAI", "US", [("greenhouse", "openai")]),
    Candidate("plaid", "Plaid", "US", [("greenhouse", "plaid")]),
    Candidate("mixpanel", "Mixpanel", "US", [("greenhouse", "mixpanel")]),
    Candidate("gusto", "Gusto", "US", [("greenhouse", "gusto")]),
    Candidate("opendoor", "Opendoor", "US", [("greenhouse", "opendoor")]),
    Candidate("vercel", "Vercel", "US", [("greenhouse", "vercel")]),
    Candidate("cloudflare", "Cloudflare", "US", [("greenhouse", "cloudflare")]),
    Candidate("anduril", "Anduril", "US", [("greenhouse", "andurilindustries")]),
    Candidate("scale", "Scale AI", "US", [("greenhouse", "scaleai")]),
    Candidate("figma", "Figma", "US", [("greenhouse", "figma")]),
    Candidate("segment", "Segment", "US", [("greenhouse", "segmentio")]),
    Candidate("gitlab", "GitLab", "US", [("greenhouse", "gitlab")]),
    Candidate("asana", "Asana", "US", [("greenhouse", "asana")]),
    Candidate("airtable", "Airtable", "US", [("greenhouse", "airtable")]),
    Candidate("discord", "Discord", "US", [("greenhouse", "discord")]),
    Candidate("chime", "Chime", "US", [("greenhouse", "chime")]),
    Candidate("lyft", "Lyft", "US", [("greenhouse", "lyft")]),
    Candidate("pinterest", "Pinterest", "US", [("greenhouse", "pinterest")]),
    Candidate("faire", "Faire", "US", [("greenhouse", "faire")]),
    Candidate("palantir", "Palantir", "US", [("lever", "palantir")]),
    Candidate("linear", "Linear", "US", [("ashby", "Linear")]),
    Candidate("vanta", "Vanta", "US", [("ashby", "Vanta")]),
    Candidate("cohere", "Cohere", "CA", [("ashby", "cohere")]),
    Candidate("modal", "Modal Labs", "US", [("ashby", "modal")]),
    Candidate("posthog", "PostHog", "US", [("ashby", "posthog")]),
    Candidate("warp", "Warp", "US", [("ashby", "warp")]),
    # ── Fintech / quant ───────────────────────────────────────────────────
    Candidate("affirm", "Affirm", "US", [("greenhouse", "affirm")]),
    Candidate("klarna", "Klarna", "US", [("greenhouse", "klarna"), ("greenhouse", "klarnapublic")]),
    Candidate("carta", "Carta", "US", [("greenhouse", "carta")]),
    Candidate(
        "mercury",
        "Mercury",
        "US",
        [("greenhouse", "mercury"), ("greenhouse", "mercurytechnologies")],
    ),
    Candidate("marqeta", "Marqeta", "US", [("greenhouse", "marqeta")]),
    Candidate(
        "two_sigma",
        "Two Sigma",
        "US",
        [("greenhouse", "twosigma"), ("greenhouse", "twosigmainvestments")],
    ),
    Candidate(
        "jane_street",
        "Jane Street",
        "US",
        [("greenhouse", "janestreet"), ("greenhouse", "janestreetcapital")],
    ),
    Candidate(
        "hrt",
        "Hudson River Trading",
        "US",
        [("greenhouse", "hudsonrivertrading"), ("greenhouse", "hrt")],
    ),
    Candidate(
        "citadel_securities", "Citadel Securities", "US", [("greenhouse", "citadelsecurities")]
    ),
    Candidate("databricks", "Databricks", "US", [("greenhouse", "databricks")]),
    Candidate("block", "Block", "US", [("greenhouse", "block"), ("greenhouse", "square")]),
    # ── AI / ML pure-play ─────────────────────────────────────────────────
    Candidate(
        "huggingface",
        "Hugging Face",
        "US",
        [("greenhouse", "huggingface"), ("ashby", "huggingface")],
    ),
    Candidate(
        "perplexity", "Perplexity AI", "US", [("greenhouse", "perplexity"), ("ashby", "perplexity")]
    ),
    Candidate(
        "character_ai",
        "Character.AI",
        "US",
        [("greenhouse", "characterai"), ("ashby", "characterai")],
    ),
    Candidate(
        "together_ai", "Together AI", "US", [("greenhouse", "togetherai"), ("ashby", "together")]
    ),
    Candidate("runway", "Runway", "US", [("greenhouse", "runwayml"), ("ashby", "runway")]),
    Candidate(
        "pinecone",
        "Pinecone",
        "US",
        [("greenhouse", "pinecone-2"), ("greenhouse", "pinecone"), ("ashby", "pinecone")],
    ),
    Candidate("wandb", "Weights & Biases", "US", [("greenhouse", "wandb"), ("ashby", "wandb")]),
    Candidate("mistral", "Mistral", "US", [("ashby", "mistral"), ("greenhouse", "mistral")]),
    # ── Biotech / health-tech ─────────────────────────────────────────────
    Candidate(
        "recursion",
        "Recursion",
        "US",
        [("greenhouse", "recursion"), ("greenhouse", "recursionpharmaceuticals")],
    ),
    Candidate("twentythreeandme", "23andMe", "US", [("greenhouse", "23andme")]),
    Candidate("verily", "Verily", "US", [("greenhouse", "verily")]),
    Candidate("insitro", "Insitro", "US", [("greenhouse", "insitro")]),
    Candidate(
        "tempus",
        "Tempus AI",
        "US",
        [("greenhouse", "tempus"), ("greenhouse", "tempusai"), ("greenhouse", "tempuslabsinc")],
    ),
    Candidate(
        "oscar_health",
        "Oscar Health",
        "US",
        [("greenhouse", "hioscar"), ("greenhouse", "oscarhealth")],
    ),
    Candidate("hims", "Hims & Hers", "US", [("greenhouse", "hims"), ("greenhouse", "himsandhers")]),
    Candidate("hinge_health", "Hinge Health", "US", [("greenhouse", "hingehealth")]),
    # ── E-commerce / marketplaces ─────────────────────────────────────────
    Candidate("etsy", "Etsy", "US", [("greenhouse", "etsy")]),
    Candidate("wayfair", "Wayfair", "US", [("greenhouse", "wayfair")]),
    Candidate("stockx", "StockX", "US", [("greenhouse", "stockx")]),
    Candidate("whatnot", "Whatnot", "US", [("greenhouse", "whatnot"), ("ashby", "whatnot")]),
    # ── Self-driving / robotics / aerospace ───────────────────────────────
    Candidate(
        "aurora", "Aurora", "US", [("greenhouse", "aurora-innovation"), ("greenhouse", "aurora")]
    ),
    Candidate(
        "zipline", "Zipline", "US", [("greenhouse", "flyzipline"), ("greenhouse", "zipline")]
    ),
    Candidate("skydio", "Skydio", "US", [("greenhouse", "skydio")]),
    Candidate("spacex", "SpaceX", "US", [("greenhouse", "spacex")]),
    Candidate("boom", "Boom Supersonic", "US", [("greenhouse", "boomsupersonic")]),
    # ── Cybersecurity ─────────────────────────────────────────────────────
    Candidate(
        "wiz", "Wiz", "US", [("greenhouse", "wiz-1"), ("greenhouse", "wiz"), ("ashby", "wiz")]
    ),
    Candidate("snyk", "Snyk", "US", [("greenhouse", "snyk")]),
    # ── Streaming / media / consumer ──────────────────────────────────────
    Candidate("spotify", "Spotify", "US", [("greenhouse", "spotify")]),
    Candidate("roblox", "Roblox", "US", [("greenhouse", "roblox")]),
    Candidate("reddit", "Reddit", "US", [("greenhouse", "reddit"), ("greenhouse", "redditinc")]),
    Candidate(
        "doordash",
        "DoorDash",
        "US",
        [("greenhouse", "doordash"), ("greenhouse", "doordashcareers")],
    ),
    # ── Dev tools / SaaS ──────────────────────────────────────────────────
    Candidate("zapier", "Zapier", "US", [("greenhouse", "zapier"), ("ashby", "zapier")]),
    Candidate("webflow", "Webflow", "US", [("greenhouse", "webflow"), ("ashby", "webflow")]),
    Candidate("pagerduty", "PagerDuty", "US", [("greenhouse", "pagerduty")]),
    Candidate("confluent", "Confluent", "US", [("greenhouse", "confluent")]),
    Candidate("elastic", "Elastic", "US", [("greenhouse", "elastic")]),
    Candidate("zendesk", "Zendesk", "US", [("greenhouse", "zendesk")]),
    Candidate("box", "Box", "US", [("greenhouse", "box")]),
    Candidate(
        "notion",
        "Notion",
        "US",
        [("greenhouse", "notion"), ("ashby", "notion"), ("ashby", "notionhq")],
    ),
    Candidate(
        "coinbase",
        "Coinbase",
        "US",
        [("greenhouse", "coinbase"), ("greenhouse", "coinbasecareers")],
    ),
    # ── Canadian (rotating handle guesses) ────────────────────────────────
    Candidate(
        "shopify", "Shopify", "CA", [("greenhouse", "shopify"), ("smartrecruiters", "shopify")]
    ),
    Candidate(
        "wealthsimple",
        "Wealthsimple",
        "CA",
        [("lever", "wealthsimple"), ("greenhouse", "wealthsimple"), ("ashby", "wealthsimple")],
    ),
    Candidate(
        "1password",
        "1Password",
        "CA",
        [("lever", "1password"), ("greenhouse", "1password"), ("ashby", "1password")],
    ),
    Candidate(
        "lightspeed",
        "Lightspeed Commerce",
        "CA",
        [
            ("greenhouse", "lightspeedhq"),
            ("greenhouse", "lightspeed"),
            ("greenhouse", "lightspeedcommerce"),
        ],
    ),
    Candidate("d2l", "D2L", "CA", [("greenhouse", "d2l"), ("lever", "d2l"), ("ashby", "d2l")]),
    Candidate(
        "tophat",
        "Top Hat",
        "CA",
        [("greenhouse", "tophat"), ("lever", "tophat"), ("ashby", "tophat")],
    ),
    Candidate("ada", "Ada", "CA", [("ashby", "Ada"), ("greenhouse", "ada"), ("lever", "ada")]),
    Candidate("hopper", "Hopper", "CA", [("greenhouse", "hopper"), ("greenhouse", "hopperinc")]),
    Candidate(
        "applyboard", "ApplyBoard", "CA", [("greenhouse", "applyboard"), ("lever", "applyboard")]
    ),
    Candidate(
        "trulioo",
        "Trulioo",
        "CA",
        [("greenhouse", "trulioo"), ("lever", "trulioo"), ("ashby", "trulioo")],
    ),
    Candidate(
        "shakepay",
        "Shakepay",
        "CA",
        [("ashby", "shakepay"), ("greenhouse", "shakepay"), ("lever", "shakepay")],
    ),
    Candidate(
        "bench",
        "Bench Accounting",
        "CA",
        [("greenhouse", "bench"), ("greenhouse", "benchaccounting"), ("lever", "bench")],
    ),
    Candidate("clio", "Clio", "CA", [("lever", "clio"), ("greenhouse", "clio"), ("ashby", "clio")]),
]


async def probe_one(client: httpx.AsyncClient, provider: str, handle: str) -> int:
    if provider == "greenhouse":
        url = GREENHOUSE.format(handle=handle)
    elif provider == "lever":
        url = LEVER.format(handle=handle)
    elif provider == "ashby":
        url = ASHBY.format(handle=handle)
    else:
        return -1
    try:
        r = await client.get(url, timeout=20.0)
    except httpx.HTTPError:
        return -1
    if r.status_code != 200:
        return 0
    try:
        data = r.json()
    except ValueError:
        return 0
    if provider == "lever":
        return len(data) if isinstance(data, list) else 0
    return len(data.get("jobs", []))


async def probe_candidate(client: httpx.AsyncClient, c: Candidate) -> dict:
    for provider, handle in c.options:
        count = await probe_one(client, provider, handle)
        if count > 0:
            return {
                "slug": c.slug,
                "name": c.name,
                "provider": provider,
                "handle": handle,
                "default_country": c.default_country,
                "jobs": count,
                "status": "ok",
            }
    return {"slug": c.slug, "name": c.name, "default_country": c.default_country, "status": "miss"}


async def run(out_path: Path | None) -> int:
    async with httpx.AsyncClient(
        headers={"User-Agent": "na-tech-jobs/probe", "Accept": "application/json"},
        follow_redirects=True,
    ) as client:
        results = await asyncio.gather(*(probe_candidate(client, c) for c in CANDIDATES))

    ok = [r for r in results if r["status"] == "ok"]
    miss = [r for r in results if r["status"] == "miss"]
    ok.sort(key=lambda r: (r["provider"], -r["jobs"]))

    print(f"# probed {len(results)} candidates :: {len(ok)} live, {len(miss)} miss\n")
    yaml_lines = ["companies:"]
    for r in ok:
        yaml_lines.append(
            "  - {{ slug: {slug}, name: {name}, provider: {provider}, handle: {handle}, default_country: {country} }}  # {jobs} jobs".format(
                slug=r["slug"],
                name=r["name"],
                provider=r["provider"],
                handle=r["handle"],
                country=r["default_country"],
                jobs=r["jobs"],
            )
        )
    if miss:
        yaml_lines.append("\n# misses (no live handle found across guesses):")
        for r in miss:
            yaml_lines.append(f"#   {r['slug']:30s} {r['name']}")

    out_text = "\n".join(yaml_lines) + "\n"
    print(out_text)
    if out_path:
        out_path.write_text(out_text)
        print(f"# wrote {out_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", default=None, help="Write probed YAML to this path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(asyncio.run(run(Path(args.output)) if args.output else run(None)))
