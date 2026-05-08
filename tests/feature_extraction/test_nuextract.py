"""Tests for the NuExtract Tier 2 wrapper.

The model itself is never loaded in CI — we mock ``_generate`` and verify
the schema-construction, parsing, coercion, and validation logic.
"""

from __future__ import annotations

import pytest

from ingestion.feature_extraction.llm import nuextract
from ingestion.feature_extraction.llm.nuextract import (
    NuExtract,
    _build_schema,
    _coerce_field,
    _parse_bool,
    _parse_int,
    _parse_output,
)


def test_build_schema_filters_to_eligible():
    schema = _build_schema(["min_years_experience", "tech_stack", "salary_min", "made_up_field"])
    assert "min_years_experience" in schema
    assert "tech_stack" in schema
    assert "salary_min" not in schema  # not LLM-eligible
    assert "made_up_field" not in schema


def test_parse_output_typical():
    raw = (
        "<|input|>...<|output|>\n"
        '{"min_years_experience": "5", "tech_stack": ["Python", "AWS"]}\n'
        "<|end-output|>"
    )
    parsed = _parse_output(raw)
    assert parsed == {"min_years_experience": "5", "tech_stack": ["Python", "AWS"]}


def test_parse_output_no_end_tag():
    raw = '<|output|>\n{"min_years_experience": "3"}'
    parsed = _parse_output(raw)
    assert parsed == {"min_years_experience": "3"}


def test_parse_output_invalid_json_returns_empty():
    raw = "<|output|>\nnot json at all\n<|end-output|>"
    assert _parse_output(raw) == {}


@pytest.mark.parametrize(
    "raw,expected",
    [("true", True), ("false", False), ("Yes", True), ("no", False), ("maybe", None), ("", None)],
)
def test_parse_bool(raw, expected):
    assert _parse_bool(raw) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [("5", 5), ("at least 7", 7), ("ten", None), ("", None), (None, None)],
)
def test_parse_int(raw, expected):
    assert _parse_int(raw) == expected


def test_coerce_int_field():
    assert _coerce_field("min_years_experience", "5") == 5
    assert _coerce_field("min_years_experience", "") is None


def test_coerce_bool_field():
    assert _coerce_field("requires_security_clearance", "true") is True
    assert _coerce_field("requires_security_clearance", "false") is False
    assert _coerce_field("requires_security_clearance", "") is None


def test_coerce_enum_valid_values():
    assert _coerce_field("min_education", "bachelors") == "bachelors"
    assert _coerce_field("min_education", "masters") == "masters"
    assert _coerce_field("clearance_level", "top_secret") == "top_secret"


def test_coerce_enum_paraphrase_fixups():
    """Common LLM paraphrases should normalize to the canonical enum value."""
    assert _coerce_field("min_education", "bachelor") == "bachelors"
    assert _coerce_field("min_education", "Doctorate") == "phd"
    assert _coerce_field("clearance_level", "TS/SCI") == "ts_sci"
    assert _coerce_field("remote_policy_extracted", "remote_north_america") == "remote-na"


def test_coerce_enum_rejects_unknown():
    """Values outside the enum set are dropped, not passed through."""
    assert _coerce_field("min_education", "intermediate") is None
    assert _coerce_field("offers_visa_sponsorship", "maybe?") is None


def test_coerce_list_drops_empty_items():
    assert _coerce_field("tech_stack", ["Python", "", "AWS"]) == ["Python", "AWS"]
    assert _coerce_field("tech_stack", ["", "", ""]) is None
    assert _coerce_field("tech_stack", []) is None


def test_coerce_list_accepts_string_singleton():
    """If NuExtract emits a string for a list field, wrap it."""
    assert _coerce_field("tech_stack", "Python") == ["Python"]


def test_coerce_string_field_team():
    assert _coerce_field("team_or_department", "Field Engineering") == "Field Engineering"
    assert _coerce_field("team_or_department", "") is None


def test_run_no_loaded_model_returns_empty(monkeypatch):
    """If transformers isn't installed, run() must degrade silently."""
    extractor = NuExtract()
    monkeypatch.setattr(extractor, "_ensure_loaded", lambda: False)
    out = extractor.run("Some description.", "Engineer", ["min_years_experience"])
    assert out == {}


def test_run_no_missing_fields_short_circuits():
    extractor = NuExtract()
    # Even without loading anything, an empty missing list should bail early.
    out = extractor.run("Some description.", "Engineer", [])
    assert out == {}
    out = extractor.run("Some description.", "Engineer", ["salary_min"])  # ineligible
    assert out == {}


def test_run_full_path_with_mocked_generate(monkeypatch):
    """End-to-end through run() with the batched generator stubbed."""

    extractor = NuExtract()
    monkeypatch.setattr(extractor, "_ensure_loaded", lambda: True)

    def fake_generate_batch(prompts: list[str]) -> list[str]:
        # Simulate NuExtract returning a clean output block per prompt.
        return [
            "<|output|>\n"
            '{"min_years_experience": "5",'
            ' "min_education": "bachelors",'
            ' "tech_stack": ["Python", "AWS"],'
            ' "team_or_department": "Field Engineering"}\n'
            "<|end-output|>"
            for _ in prompts
        ]

    monkeypatch.setattr(extractor, "_generate_batch", fake_generate_batch)

    out = extractor.run(
        "We are hiring an engineer with 5 years experience.",
        title="Senior Engineer",
        missing_fields=[
            "min_years_experience",
            "min_education",
            "tech_stack",
            "team_or_department",
            "offers_visa_sponsorship",  # absent in mock output
        ],
    )
    assert out["min_years_experience"].value == 5
    assert out["min_years_experience"].source == "llm"
    assert out["min_education"].value == "bachelors"
    assert out["tech_stack"].value == ["Python", "AWS"]
    assert out["team_or_department"].value == "Field Engineering"
    assert "offers_visa_sponsorship" not in out


def test_run_batch_aligns_outputs_with_inputs(monkeypatch):
    """run_batch must keep result indexes in sync with input indexes,
    even when some items short-circuit (empty schema, blank text)."""
    extractor = NuExtract()
    monkeypatch.setattr(extractor, "_ensure_loaded", lambda: True)

    def fake_generate_batch(prompts: list[str]) -> list[str]:
        return ['<|output|>\n{"team_or_department": "Eng"}\n<|end-output|>' for _ in prompts]

    monkeypatch.setattr(extractor, "_generate_batch", fake_generate_batch)

    items = [
        ("Some real description.", "Engineer", ["team_or_department"]),
        ("", "Empty desc", ["team_or_department"]),  # blank text → skip
        ("Another description.", "Director", ["salary_min"]),  # no eligible fields → skip
        ("Final description.", "Scientist", ["team_or_department"]),
    ]
    out = extractor.run_batch(items)
    assert out[0]["team_or_department"].value == "Eng"
    assert out[1] == {}
    assert out[2] == {}
    assert out[3]["team_or_department"].value == "Eng"


def test_run_handles_generation_failure(monkeypatch, caplog):
    """If generate raises, run() returns {} and logs the failure."""
    extractor = NuExtract()
    monkeypatch.setattr(extractor, "_ensure_loaded", lambda: True)

    def boom(prompts: list[str]) -> list[str]:
        raise RuntimeError("OOM")

    monkeypatch.setattr(extractor, "_generate_batch", boom)
    with caplog.at_level("WARNING", logger="feature_extraction.nuextract"):
        out = extractor.run("text", "title", ["min_years_experience"])
    assert out == {}
    assert any("NuExtract batch generation failed" in r.message for r in caplog.records)


def test_module_constants_aligned():
    """Sanity: every LLM-eligible field has a schema entry and vice versa."""
    from ingestion.feature_extraction.cascade import LLM_ELIGIBLE_FIELDS

    schema_fields = set(nuextract.LLM_FIELD_SCHEMAS)
    eligible = set(LLM_ELIGIBLE_FIELDS)
    # Every cascade-eligible field should have a schema entry.
    assert eligible.issubset(schema_fields), eligible - schema_fields
