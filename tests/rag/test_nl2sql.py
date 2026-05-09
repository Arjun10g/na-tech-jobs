"""Tests for rag.nl2sql.

The safety layer is the critical piece — every reasonable attack surface
gets a test (DDL keywords, table escape, column escape, multi-statement,
non-SELECT). Execution is tested against a small in-memory parquet so we
don't need the full curated corpus.

LLM call paths are tested with a MockLLM — no live API calls.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rag.nl2sql import (
    DENYLISTED_KEYWORDS,
    MockLLM,
    NL2SQLResult,
    SQLExecutionError,
    SQLSafetyError,
    execute_sql,
    nl_to_sql,
    schema_description,
    validate_sql,
)

# ── Fixture: tiny parquet with the expected schema ────────────────────────


@pytest.fixture
def tiny_parquet(tmp_path) -> Path:
    df = pd.DataFrame(
        [
            {
                "id": "j1",
                "title": "Senior MLE",
                "company_name": "Acme",
                "country": "US",
                "region": "CA",
                "city": "San Francisco",
                "remote_policy": "hybrid",
                "source": "greenhouse",
                "salary_min_usd_yearly": 180_000.0,
                "salary_max_usd_yearly": 240_000.0,
                "salary_disclosed": True,
                "seniority_label_v1": "senior",
                "role_family_v1": "MLE",
                "predicted_salary_usd_v1": 210_000.0,
                "extracted_skills_v1": ["Python", "PyTorch"],
            },
            {
                "id": "j2",
                "title": "Staff DE",
                "company_name": "Beta Co",
                "country": "CA",
                "region": "ON",
                "city": "Toronto",
                "remote_policy": "remote",
                "source": "lever",
                "salary_min_usd_yearly": 200_000.0,
                "salary_max_usd_yearly": 260_000.0,
                "salary_disclosed": True,
                "seniority_label_v1": "staff",
                "role_family_v1": "DE",
                "predicted_salary_usd_v1": 230_000.0,
                "extracted_skills_v1": ["dbt", "Snowflake"],
            },
            {
                "id": "j3",
                "title": "Senior DS",
                "company_name": "Gamma",
                "country": "US",
                "region": "NY",
                "city": "New York",
                "remote_policy": "onsite",
                "source": "ashby",
                "salary_min_usd_yearly": None,
                "salary_max_usd_yearly": None,
                "salary_disclosed": False,
                "seniority_label_v1": "senior",
                "role_family_v1": "DS",
                "predicted_salary_usd_v1": 195_000.0,
                "extracted_skills_v1": ["SQL"],
            },
        ]
    )
    p = tmp_path / "jobs.parquet"
    df.to_parquet(p)
    return p


# ── Schema description ────────────────────────────────────────────────────


def test_schema_description_lists_all_columns():
    desc = schema_description()
    for col in (
        "id",
        "title",
        "country",
        "salary_max_usd_yearly",
        "seniority_label_v1",
        "role_family_v1",
        "predicted_salary_usd_v1",
        "extracted_skills_v1",
    ):
        assert col in desc


# ── Safety: happy paths ───────────────────────────────────────────────────


def test_validate_select_simple_passes():
    sql = "SELECT title, country FROM jobs WHERE country = 'US'"
    tree = validate_sql(sql)
    assert tree is not None


def test_validate_select_with_aggregate_passes():
    sql = (
        "SELECT country, AVG(predicted_salary_usd_v1) AS avg_salary "
        "FROM jobs WHERE seniority_label_v1 = 'senior' GROUP BY country"
    )
    validate_sql(sql)


def test_validate_select_alias_referenced_in_order_by_passes():
    """Aliases from `... AS n` are legal references in ORDER BY / HAVING."""
    sql = "SELECT country, COUNT(*) AS n FROM jobs GROUP BY country ORDER BY n DESC"
    validate_sql(sql)


def test_validate_select_alias_referenced_in_having_passes():
    sql = "SELECT country, COUNT(*) AS n FROM jobs GROUP BY country HAVING n > 5"
    validate_sql(sql)


def test_validate_select_with_cte_passes():
    sql = (
        "WITH senior_us AS (SELECT * FROM jobs WHERE country = 'US' "
        "AND seniority_label_v1 = 'senior') "
        "SELECT role_family_v1, COUNT(*) FROM senior_us GROUP BY role_family_v1"
    )
    validate_sql(sql)


def test_validate_select_star_passes():
    sql = "SELECT * FROM jobs LIMIT 10"
    validate_sql(sql)


def test_validate_subquery_passes():
    sql = (
        "SELECT t.country, t.role_family_v1 FROM "
        "(SELECT country, role_family_v1 FROM jobs WHERE salary_disclosed) AS t"
    )
    validate_sql(sql)


def test_validate_trailing_semicolon_ok():
    sql = "SELECT title FROM jobs;"
    validate_sql(sql)


# ── Safety: rejections ────────────────────────────────────────────────────


def test_validate_empty_rejected():
    with pytest.raises(SQLSafetyError, match="empty"):
        validate_sql("")
    with pytest.raises(SQLSafetyError, match="empty"):
        validate_sql("   ")


def test_validate_insert_rejected():
    with pytest.raises(SQLSafetyError):
        validate_sql("INSERT INTO jobs (id) VALUES ('xxx')")


def test_validate_update_rejected():
    with pytest.raises(SQLSafetyError):
        validate_sql("UPDATE jobs SET title = 'x' WHERE id = 'j1'")


def test_validate_delete_rejected():
    with pytest.raises(SQLSafetyError):
        validate_sql("DELETE FROM jobs WHERE id = 'j1'")


def test_validate_drop_rejected():
    with pytest.raises(SQLSafetyError):
        validate_sql("DROP TABLE jobs")


def test_validate_create_rejected():
    with pytest.raises(SQLSafetyError):
        validate_sql("CREATE TABLE foo AS SELECT * FROM jobs")


def test_validate_alter_rejected():
    with pytest.raises(SQLSafetyError):
        validate_sql("ALTER TABLE jobs ADD COLUMN x INT")


def test_validate_attach_rejected():
    with pytest.raises(SQLSafetyError):
        validate_sql("ATTACH DATABASE ':memory:' AS sneaky")


def test_validate_pragma_rejected():
    with pytest.raises(SQLSafetyError):
        validate_sql("PRAGMA foreign_keys = OFF")


def test_validate_copy_rejected():
    with pytest.raises(SQLSafetyError):
        validate_sql("COPY jobs TO '/tmp/leak.csv'")


def test_validate_install_load_rejected():
    with pytest.raises(SQLSafetyError):
        validate_sql("INSTALL httpfs")
    with pytest.raises(SQLSafetyError):
        validate_sql("LOAD httpfs")


def test_validate_multi_statement_rejected():
    """Compound `SELECT ...; DELETE ...` must not slip past the parser."""
    with pytest.raises(SQLSafetyError):
        validate_sql("SELECT 1; DELETE FROM jobs")


def test_validate_disallowed_table_rejected():
    with pytest.raises(SQLSafetyError, match="table"):
        validate_sql("SELECT * FROM users")


def test_validate_disallowed_table_via_join_rejected():
    with pytest.raises(SQLSafetyError, match="table"):
        validate_sql("SELECT j.id FROM jobs j JOIN secret_table s ON s.id = j.id")


def test_validate_disallowed_column_rejected():
    with pytest.raises(SQLSafetyError, match="column"):
        validate_sql("SELECT password FROM jobs")


def test_validate_disallowed_column_in_where_rejected():
    with pytest.raises(SQLSafetyError, match="column"):
        validate_sql("SELECT title FROM jobs WHERE secret_field = 'x'")


@pytest.mark.parametrize("kw", [k for k in DENYLISTED_KEYWORDS if k.strip()])
def test_each_denylisted_keyword_caught(kw):
    """Smoke-test that every keyword in DENYLISTED_KEYWORDS triggers."""
    sql = f"SELECT * FROM jobs WHERE country = '{kw} GO'"  # only string content
    # The keyword in a string literal can be a false positive — we accept
    # that for safety. The point is no keyword sneaks past the prefilter.
    with pytest.raises(SQLSafetyError):
        validate_sql(sql)


def test_string_literal_with_keyword_in_it_is_rejected_safely():
    """Conservative: even keywords-in-strings are rejected. The LLM should
    not be producing them; this protects against literal injection."""
    sql = "SELECT title FROM jobs WHERE company_name = 'DROP IT INC'"
    with pytest.raises(SQLSafetyError):
        validate_sql(sql)


# ── Execution ─────────────────────────────────────────────────────────────


def test_execute_simple_select(tiny_parquet):
    df = execute_sql("SELECT id, title FROM jobs ORDER BY id", tiny_parquet)
    assert list(df["id"]) == ["j1", "j2", "j3"]


def test_execute_filter_and_aggregate(tiny_parquet):
    df = execute_sql(
        "SELECT country, COUNT(*) AS n FROM jobs GROUP BY country ORDER BY country",
        tiny_parquet,
    )
    assert dict(zip(df["country"], df["n"], strict=True)) == {"CA": 1, "US": 2}


def test_execute_caps_rows(tiny_parquet):
    df = execute_sql("SELECT * FROM jobs", tiny_parquet, max_rows=2)
    assert len(df) == 2


def test_execute_invalid_sql_raises(tiny_parquet):
    with pytest.raises(SQLExecutionError):
        execute_sql("SELECT no_such_function(*) FROM jobs", tiny_parquet)


# ── End-to-end with MockLLM ───────────────────────────────────────────────


def test_nl_to_sql_happy_path(tiny_parquet):
    sql = "SELECT country, COUNT(*) AS n FROM jobs GROUP BY country"
    llm = MockLLM(response=sql)
    result = nl_to_sql("how many jobs per country", tiny_parquet, llm=llm)
    assert result.error is None
    assert result.sql == sql
    assert result.n_rows == 2


def test_nl_to_sql_strips_markdown_fences(tiny_parquet):
    fenced = "```sql\nSELECT id FROM jobs\n```"
    llm = MockLLM(response=fenced)
    result = nl_to_sql("ids", tiny_parquet, llm=llm)
    assert result.error is None
    assert result.sql == "SELECT id FROM jobs"


def test_nl_to_sql_rejects_unsafe(tiny_parquet):
    llm = MockLLM(response="DROP TABLE jobs")
    result = nl_to_sql("delete it all", tiny_parquet, llm=llm)
    assert result.error is not None
    assert "rejected" in result.error.lower()
    assert result.rows is None


def test_nl_to_sql_returns_error_on_llm_failure(tiny_parquet):
    class BoomLLM(MockLLM):
        def generate(self, system, user, *, max_tokens=512):  # noqa: ARG002
            raise RuntimeError("api blew up")

    result = nl_to_sql("anything", tiny_parquet, llm=BoomLLM())
    assert result.error and "LLM call failed" in result.error
    assert result.sql is None


def test_nl_to_sql_returns_error_on_execution_failure(tiny_parquet):
    # Validates fine (`title` is allowed) but DuckDB will fail because the
    # function doesn't exist — we should surface "Execution failed".
    llm = MockLLM(response="SELECT no_such_fn(title) FROM jobs")
    result = nl_to_sql("anything", tiny_parquet, llm=llm)
    assert result.error and "Execution failed" in result.error


def test_nl2sql_result_serializes(tiny_parquet):
    from rag.nl2sql import serialize_result

    llm = MockLLM(response="SELECT id, title FROM jobs ORDER BY id LIMIT 2")
    result = nl_to_sql("ids", tiny_parquet, llm=llm)
    payload = serialize_result(result)
    assert payload["sql"].startswith("SELECT")
    assert payload["n_rows"] == 2
    assert isinstance(payload["rows_preview"], list)
    assert payload["rows_preview"][0]["id"] == "j1"


def test_default_llm_raises_without_creds(monkeypatch):
    from rag.nl2sql import default_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="LLM"):
        default_llm()


def test_default_llm_picks_anthropic_when_configured(monkeypatch):
    from rag.nl2sql import AnthropicLLM, default_llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("HF_TOKEN", "y")
    assert isinstance(default_llm(), AnthropicLLM)


def test_default_llm_falls_back_to_hf(monkeypatch):
    from rag.nl2sql import HFInferenceLLM, default_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "y")
    assert isinstance(default_llm(), HFInferenceLLM)


# ── NL2SQLResult dataclass ────────────────────────────────────────────────


def test_nl2sql_result_construct():
    r = NL2SQLResult(question="x", sql="SELECT 1", rows=None, error=None, n_rows=0)
    assert r.question == "x"
    assert r.sql == "SELECT 1"
