"""Tests for ``eda.split``: determinism, stratification, frozen split.

These tests are **load-bearing**: if `freeze_split` ever returns different
test rows for the same (df, seed), every model retraining downstream
silently changes its eval set. Lock the contract here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eda.split import freeze_split, load_test_ids, write_split


@pytest.fixture
def fake_jobs() -> pd.DataFrame:
    rows = []
    for i in range(500):
        # 70% US/greenhouse, 20% US/lever, 10% CA/ashby (matches our real ratios)
        if i < 350:
            country, source = "US", "greenhouse"
        elif i < 450:
            country, source = "US", "lever"
        else:
            country, source = "CA", "ashby"
        rows.append({"id": f"row{i:04d}", "country": country, "source": source})
    return pd.DataFrame(rows)


def test_split_is_deterministic(fake_jobs):
    a = freeze_split(fake_jobs, seed=42)
    b = freeze_split(fake_jobs, seed=42)
    assert a["test_ids"] == b["test_ids"]
    assert a["train_ids"] == b["train_ids"]


def test_split_changes_with_seed(fake_jobs):
    a = freeze_split(fake_jobs, seed=42)
    b = freeze_split(fake_jobs, seed=1)
    assert set(a["test_ids"]) != set(b["test_ids"])


def test_split_test_frac_within_tolerance(fake_jobs):
    splits = freeze_split(fake_jobs, test_frac=0.20, seed=42)
    actual = splits["params"]["test_frac_actual"]
    assert 0.10 <= actual <= 0.30  # generous bound for n=100


def test_split_train_test_disjoint(fake_jobs):
    splits = freeze_split(fake_jobs, seed=42)
    assert set(splits["train_ids"]).isdisjoint(splits["test_ids"])
    total = len(splits["train_ids"]) + len(splits["test_ids"])
    assert total == len(fake_jobs)


def test_strata_each_have_test_rows(fake_jobs):
    """Each (country, source) cell should be represented in test."""
    splits = freeze_split(fake_jobs, seed=42)
    for stratum in splits["stratum_counts"]:
        assert stratum["n_test"] >= 1, stratum


def test_persist_and_reload_roundtrip(fake_jobs, tmp_path):
    splits = freeze_split(fake_jobs, seed=42)
    p = tmp_path / "split.json"
    write_split(splits, p)
    loaded = load_test_ids(p)
    assert loaded == set(splits["test_ids"])


def test_split_misses_id_column():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(KeyError):
        freeze_split(df)
