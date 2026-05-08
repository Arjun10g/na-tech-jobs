"""End-to-end cascade tests."""

from __future__ import annotations

from ingestion.feature_extraction import extract_features


def test_cascade_full_description():
    desc = """
    ## Senior Machine Learning Engineer

    We're hiring a Senior ML Engineer to work on our recommendation systems.
    This is a hybrid role, 3 days per week in our San Francisco office.

    ## Requirements

    - 7+ years of experience building production ML systems
    - Bachelor's degree in Computer Science or related field
    - Strong Python and PyTorch experience
    - Familiarity with AWS and Kubernetes

    ## Compensation

    The annual base salary range for this role is $200,000 - $260,000 USD,
    plus RSUs and a target performance bonus.
    We offer relocation assistance.
    """
    feats = extract_features(desc, title="Senior Machine Learning Engineer")
    assert feats["salary_min"] == 200_000
    assert feats["salary_max"] == 260_000
    assert feats["salary_currency"] == "USD"
    assert feats["salary_period"] == "year"
    assert feats["salary_disclosed"] is True
    assert feats["min_years_experience"] == 7
    assert feats["min_education"] == "bachelors"
    assert feats["remote_policy_extracted"] == "hybrid"
    assert feats["offers_equity"] is True
    assert feats["equity_form"] == "rsu"
    assert feats["bonus_mentioned"] is True
    assert feats["bonus_type"] == "performance"
    assert feats["offers_relocation"] is True
    assert "Python" in feats["tech_stack"]
    assert "PyTorch" in feats["tech_stack"]
    assert "AWS" in feats["tech_stack"]
    assert "Kubernetes" in feats["tech_stack"]
    # Provenance recorded
    meta = feats["extraction_meta"]
    assert meta["salary_min"]["source"] == "regex"
    assert meta["salary_min"]["confidence"] >= 0.7


def test_cascade_minimal_description():
    feats = extract_features("Generic description with no notable signals.", title="")
    # Defaults: posting_quality=real, otherwise mostly empty
    assert feats["posting_quality"] == "real"
    assert "salary_disclosed" not in feats or feats.get("salary_disclosed") is None


def test_cascade_evergreen_posting_flagged():
    feats = extract_features(
        "Hi! Tell us about your background.",
        title="Future Opportunities - Join Our Talent Network",
    )
    assert feats["posting_quality"] == "evergreen_pool"


def test_cascade_extraction_meta_present():
    feats = extract_features("5+ years required.", title="Engineer")
    assert "extraction_meta" in feats
    assert "extraction_version" in feats
    assert feats["extraction_version"] == "v1"
