"""Tests for monitoring.alerts."""

from __future__ import annotations

import httpx
import pytest

from monitoring import alerts


@pytest.fixture
def webhook_capturer(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patch httpx.post so we can inspect the JSON payload without a network call."""

    captured: list[dict] = []

    class FakeResponse:
        status_code = 204

        def raise_for_status(self) -> None:
            return None

    def fake_post(url, json=None, timeout=None, **_):
        captured.append({"url": url, "json": json})
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.test/hook")
    return captured


def test_discord_alert_sends_payload(webhook_capturer):
    ok = alerts.discord_alert("Hello", "world", fields={"rows": 42})
    assert ok is True
    assert len(webhook_capturer) == 1
    payload = webhook_capturer[0]["json"]
    embed = payload["embeds"][0]
    assert embed["title"] == "Hello"
    assert embed["description"] == "world"
    assert {"name": "rows", "value": "42", "inline": True} in embed["fields"]


def test_alert_success_uses_green_color(webhook_capturer):
    alerts.alert_success("All good")
    embed = webhook_capturer[0]["json"]["embeds"][0]
    assert embed["color"] == alerts.DISCORD_GREEN


def test_alert_failure_uses_red_color(webhook_capturer):
    alerts.alert_failure("Boom", "stack trace")
    embed = webhook_capturer[0]["json"]["embeds"][0]
    assert embed["color"] == alerts.DISCORD_RED


def test_no_webhook_returns_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert alerts.discord_alert("title") is False


def test_http_error_does_not_raise_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.test/hook")

    def boom(*_, **__):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx, "post", boom)
    assert alerts.discord_alert("title") is False


def test_http_error_raises_when_requested(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.test/hook")

    def boom(*_, **__):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(httpx.HTTPError):
        alerts.discord_alert("title", raise_on_error=True)
