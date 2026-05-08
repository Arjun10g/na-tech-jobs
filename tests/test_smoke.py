"""Smoke tests proving the package imports and the Gradio app builds."""

from __future__ import annotations

import gradio as gr

from app.main import build_app, status


def test_status_string_mentions_project_name() -> None:
    body = status()
    assert "na-tech-jobs" in body


def test_build_app_returns_blocks() -> None:
    app = build_app()
    assert isinstance(app, gr.Blocks)
