"""Tests for the rest of the Tier 1 regex extractors."""

from __future__ import annotations

import pytest

from ingestion.feature_extraction.regex import (
    comp_extras,
    contract_quality,
    experience_education,
    remote_schedule,
    requirements,
    tech_stack,
)

# ── experience + education ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected_min,expected_max",
    [
        ("Requires 5+ years of experience.", 5, None),
        ("3-5 years of experience required.", 3, 5),
        ("At least 7 years of relevant experience.", 7, None),
        ("Minimum of 10 years in the field.", 10, None),
    ],
)
def test_years_extraction(text, expected_min, expected_max):
    out = experience_education.run(text)
    assert out["min_years_experience"].value == expected_min
    if expected_max is not None:
        assert out["max_years_experience"].value == expected_max


def test_years_no_match():
    assert "min_years_experience" not in experience_education.run("We are a fast-growing team.")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Bachelor's degree required.", "bachelors"),
        ("Master's degree in Computer Science.", "masters"),
        ("PhD or equivalent experience.", "phd"),
        ("BSc in Engineering.", "bachelors"),
        ("MBA preferred.", "masters"),
        ("High school diploma or GED required.", "high_school"),
    ],
)
def test_education_extraction(text, expected):
    out = experience_education.run(text)
    assert out["min_education"].value == expected


def test_education_no_false_positive_on_bare_ms():
    """Bare 'MS' or 'MA' shouldn't trigger masters detection."""
    text = "MS Office expertise required. Teams in MA office."
    assert "min_education" not in experience_education.run(text)


# ── requirements ──────────────────────────────────────────────────────────


def test_security_clearance_ts_sci():
    out = requirements.run("Active TS/SCI clearance required.")
    assert out["requires_security_clearance"].value is True
    assert out["clearance_level"].value == "ts_sci"


def test_security_clearance_secret():
    out = requirements.run("Must hold an active Secret clearance.")
    assert out["requires_security_clearance"].value is True
    assert out["clearance_level"].value == "secret"


def test_no_clearance_required_negation():
    """When the text says 'clearance not required', we skip the bool."""
    out = requirements.run("Top Secret clearance is not required for this role.")
    assert "requires_security_clearance" not in out


def test_us_citizenship_required():
    out = requirements.run("Must be a US citizen due to ITAR regulations.")
    assert out["requires_citizenship"].value == ["US"]


def test_canadian_citizenship_required():
    out = requirements.run("Must be a Canadian citizen or permanent resident.")
    assert out["requires_citizenship"].value == ["CA"]


def test_visa_sponsorship_yes():
    out = requirements.run("We offer visa sponsorship for qualified candidates.")
    assert out["offers_visa_sponsorship"].value == "yes"


def test_visa_sponsorship_no():
    out = requirements.run("Must be authorized to work in the US without sponsorship.")
    assert out["offers_visa_sponsorship"].value == "no"


# ── remote / schedule ─────────────────────────────────────────────────────


def test_remote_policy_remote():
    out = remote_schedule.run("This is a fully remote position.")
    assert out["remote_policy_extracted"].value == "remote"


def test_remote_policy_hybrid():
    out = remote_schedule.run("Hybrid role: 3 days per week in our SF office.")
    assert out["remote_policy_extracted"].value == "hybrid"


def test_remote_policy_onsite():
    out = remote_schedule.run("This is an onsite role only. No remote work.")
    assert out["remote_policy_extracted"].value == "onsite"


def test_on_call_required():
    out = remote_schedule.run("Participate in an on-call rotation.")
    assert out["on_call_required"].value is True


def test_no_on_call():
    out = remote_schedule.run("There is no on-call requirement for this position.")
    assert out["on_call_required"].value is False


def test_travel_percent_with_travel_word():
    out = remote_schedule.run("Travel up to 25% of the time required.")
    assert out["max_travel_percent"].value == 25


def test_no_false_positive_on_401k_match():
    """Critical regression: 'up to 100% match on 401k' must not trip travel."""
    out = remote_schedule.run("We offer up to 100% match on your 401(k).")
    assert "max_travel_percent" not in out


# ── compensation extras ───────────────────────────────────────────────────


def test_equity_rsu():
    out = comp_extras.run("Generous RSU package included.")
    assert out["offers_equity"].value is True
    assert out["equity_form"].value == "rsu"


def test_equity_options():
    out = comp_extras.run("Stock options are part of the compensation package.")
    assert out["offers_equity"].value is True
    assert out["equity_form"].value == "options"


def test_signing_bonus():
    out = comp_extras.run("This role includes a signing bonus.")
    assert out["bonus_mentioned"].value is True
    assert out["bonus_type"].value == "signing"


def test_relocation_offered():
    out = comp_extras.run("We offer relocation assistance.")
    assert out["offers_relocation"].value is True


def test_relocation_negated():
    out = comp_extras.run("No relocation assistance is provided for this role.")
    assert out["offers_relocation"].value is False


# ── contract / quality / language / manager ──────────────────────────────


def test_contract_full_time():
    out = contract_quality.run("Full-time permanent position.", title="Senior Engineer")
    assert out["contract_type"].value == "full_time"


def test_contract_internship():
    out = contract_quality.run(
        "Summer internship opportunity for students.", title="Software Engineering Intern"
    )
    assert out["contract_type"].value == "internship"


def test_contract_contract():
    out = contract_quality.run("12-month fixed-term contract.", title="Data Engineer (FTC)")
    assert out["contract_type"].value == "contract"


def test_posting_quality_evergreen():
    out = contract_quality.run(
        "Tell us about yourself.",
        title="Future Opportunities - North Star: Calling all Canadians Abroad",
    )
    assert out["posting_quality"].value == "evergreen_pool"


def test_posting_quality_real_default():
    out = contract_quality.run("We are hiring an engineer.", title="Senior Software Engineer")
    assert out["posting_quality"].value == "real"


def test_language_french():
    out = contract_quality.run("Bilingual (English/French) required.")
    assert "fr" in out["language_requirements"].value


def test_manager_role_director():
    out = contract_quality.run("Lead the team.", title="Director of Engineering")
    assert out["manager_role"].value == "director"


def test_manager_role_only_when_signal():
    """No false 'ic' default — leave None when title is plain."""
    out = contract_quality.run("Build features.", title="Senior Software Engineer")
    assert "manager_role" not in out


def test_direct_reports_count():
    out = contract_quality.run("You will manage 8 direct reports.", title="Engineering Manager")
    assert out["direct_reports_count"].value == 8


# ── tech stack ────────────────────────────────────────────────────────────


def test_tech_stack_python_aws():
    out = tech_stack.run("We use Python on AWS with PyTorch and SQL.")
    assert "Python" in out["tech_stack"].value
    assert "AWS" in out["tech_stack"].value
    assert "PyTorch" in out["tech_stack"].value
    assert "SQL" in out["tech_stack"].value


def test_tech_stack_no_false_r():
    """Bare letter R in prose must not match the R language."""
    out = tech_stack.run("We are a fast-paced team focused on customer success.")
    assert out == {}


def test_tech_stack_no_false_next_word():
    """The word 'next' in prose shouldn't match Next.js."""
    out = tech_stack.run("We are looking for the next big thing.")
    assert out == {}


def test_tech_stack_kubernetes_docker():
    out = tech_stack.run("Familiarity with Docker and Kubernetes (k8s) required.")
    assert "Docker" in out["tech_stack"].value
    assert "Kubernetes" in out["tech_stack"].value


def test_tech_stack_aws_case_sensitive():
    """'aws' in lowercase prose shouldn't match (case-sensitive for AWS)."""
    out = tech_stack.run("Some lowercase text mentioning aws platform incidentally.")
    # The pattern is case-sensitive for AWS, so this should not match.
    assert "AWS" not in (out.get("tech_stack").value if "tech_stack" in out else [])
