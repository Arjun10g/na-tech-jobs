"""Discord webhook alerting used across ingest, drift, and retraining pipelines."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DISCORD_GREEN = 0x57F287
DISCORD_RED = 0xED4245
DISCORD_YELLOW = 0xFEE75C
DISCORD_BLURPLE = 0x5865F2


def _resolve_webhook(webhook_url: str | None) -> str | None:
    return webhook_url or os.environ.get("DISCORD_WEBHOOK_URL") or None


def discord_alert(
    title: str,
    description: str = "",
    *,
    color: int = DISCORD_BLURPLE,
    fields: dict[str, Any] | None = None,
    webhook_url: str | None = None,
    timeout: float = 10.0,
    raise_on_error: bool = False,
) -> bool:
    """Post a single embed to the Discord webhook. Returns True on success."""
    url = _resolve_webhook(webhook_url)
    if not url:
        logger.warning("DISCORD_WEBHOOK_URL unset; skipping alert: %s", title)
        return False

    embed: dict[str, Any] = {"title": title[:256], "color": color}
    if description:
        embed["description"] = description[:4000]
    if fields:
        embed["fields"] = [
            {"name": str(k)[:256], "value": str(v)[:1024], "inline": True}
            for k, v in fields.items()
        ]

    payload = {"username": "na-tech-jobs", "embeds": [embed]}
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("discord webhook failed: %s", exc)
        if raise_on_error:
            raise
        return False
    return True


def alert_success(title: str, description: str = "", **fields: Any) -> bool:
    return discord_alert(title, description, color=DISCORD_GREEN, fields=fields or None)


def alert_warning(title: str, description: str = "", **fields: Any) -> bool:
    return discord_alert(title, description, color=DISCORD_YELLOW, fields=fields or None)


def alert_failure(title: str, description: str = "", **fields: Any) -> bool:
    return discord_alert(title, description, color=DISCORD_RED, fields=fields or None)
