"""Natural-language → SQL over the curated parquet, with a hard safety layer.

Per CLAUDE.md §8 and §11 the safety layer is **mandatory** — never skip it.
Pipeline:

    user question
        → LLM (prompt + schema)
        → SQL string
        → sqlglot validation (allowlist + DDL reject)
        → DuckDB execution (row + time caps)
        → results

The LLM half is intentionally pluggable (``LLMClient``) so tests can run
without a live API. Production paths: ``HFInferenceClient`` (HF Inference
API for ``Qwen/Qwen2.5-7B-Instruct``) and ``AnthropicClient`` (Claude API
when ``ANTHROPIC_API_KEY`` is set).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot
import sqlglot.expressions as exp

logger = logging.getLogger("rag.nl2sql")


# ── Schema (allowlist) ────────────────────────────────────────────────────


# The jobs table the user is allowed to query. Both raw curated and the
# Phase 4 enriched versioned columns are exposed; anything else (raw
# extractor payloads, deprecated columns) is rejected at validation time.
ALLOWED_TABLES: frozenset[str] = frozenset({"jobs"})

ALLOWED_COLUMNS: dict[str, frozenset[str]] = {
    "jobs": frozenset(
        {
            # Identity / metadata
            "id",
            "company_slug",
            "company_name",
            "title",
            "url",
            "source",
            "posted_at",
            "scraped_at",
            # Geography
            "country",
            "region",
            "city",
            "remote_policy",
            # Salary (raw + normalized)
            "salary_min",
            "salary_max",
            "salary_currency",
            "salary_period",
            "salary_min_usd_yearly",
            "salary_max_usd_yearly",
            "salary_disclosed",
            # Regex labels (weak supervision, retained for back-compat)
            "seniority_extracted",
            "role_family_extracted",
            # Phase 1b feature-extraction columns
            "min_years_experience",
            "min_education",
            "manager_role",
            "clearance_level",
            "contract_type",
            "equity_form",
            "bonus_type",
            "tech_stack",
            "requires_security_clearance",
            "offers_visa_sponsorship",
            "offers_relocation",
            "offers_equity",
            "bonus_mentioned",
            "on_call_required",
            "requires_citizenship",
            "language_requirements",
            # Phase 4 versioned predictions
            "seniority_label_v1",
            "seniority_confidence_v1",
            "role_family_v1",
            "role_family_confidence_v1",
            "predicted_salary_usd_v1",
            "extracted_skills_v1",
            "prediction_model_version",
        }
    ),
}

# Always rejected — matches DuckDB / SQL surface that lets queries escape
# the read-only contract.
DENYLISTED_KEYWORDS: tuple[str, ...] = (
    "ATTACH",
    "DETACH",
    "INSTALL",
    "LOAD",
    "PRAGMA",
    "EXPORT",
    "IMPORT",
    "COPY",
    "SET ",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "VACUUM",
    "ANALYZE",
    "REINDEX",
    "GRANT",
    "REVOKE",
)

DEFAULT_MAX_ROWS: int = 1000
DEFAULT_EXEC_SECONDS: float = 5.0


# ── Errors ────────────────────────────────────────────────────────────────


class SQLSafetyError(ValueError):
    """Raised when a generated SQL fails the safety check."""


class SQLExecutionError(RuntimeError):
    """Raised when DuckDB execution fails or hits a cap."""


# ── Schema description (LLM-facing) ───────────────────────────────────────


def schema_description() -> str:
    """Compact human/LLM-readable description of the allowed schema.

    Embedded into the LLM prompt so it knows what columns exist. Kept
    short — verbose schemas hurt prompt quality.
    """
    cols = sorted(ALLOWED_COLUMNS["jobs"])
    return (
        "Table: jobs (one row per active job posting from the curated "
        "North American tech-jobs corpus).\n\n"
        "Columns (all SELECT-able):\n  - " + "\n  - ".join(cols) + "\n\n"
        "Key columns:\n"
        "  - country: ISO alpha-2 ('US', 'CA').\n"
        "  - region: state / province code where derivable; may be null.\n"
        "  - posted_at: TIMESTAMP UTC; the job's posted_at date.\n"
        "  - salary_min_usd_yearly / salary_max_usd_yearly: numeric, "
        "USD/year, NULL when not disclosed.\n"
        "  - salary_disclosed: BOOLEAN — true iff the company disclosed "
        "the range.\n"
        "  - seniority_label_v1: classifier output, one of "
        "{intern, junior, senior, staff, principal, manager, director}.\n"
        "  - role_family_v1: classifier output, one of "
        "{AS, DA, DE, DS, MLE, RS, SWE-ML}.\n"
        "  - predicted_salary_usd_v1: float, salary regressor's USD/yr "
        "prediction (populated for every row).\n"
        "  - extracted_skills_v1: VARCHAR[] — list of canonical skill "
        "names from the regex tech_stack extractor.\n"
    )


# ── Safety layer ──────────────────────────────────────────────────────────


def _has_denylisted_keyword(sql: str) -> str | None:
    """Cheap pre-parse check — catches sneaky compound statements."""
    upper = sql.upper()
    for kw in DENYLISTED_KEYWORDS:
        # Word-boundary match so ALTER doesn't fire on "altered_at" etc.
        pattern = r"\b" + re.escape(kw.strip()) + r"\b"
        if re.search(pattern, upper):
            return kw.strip()
    return None


def validate_sql(sql: str, *, allowed_tables: frozenset[str] = ALLOWED_TABLES) -> exp.Expression:
    """Parse the SQL with sqlglot and reject if it violates the safety contract.

    Raises ``SQLSafetyError`` with a human-readable message on rejection;
    returns the parsed expression on acceptance.
    """
    if not sql or not sql.strip():
        raise SQLSafetyError("empty SQL")
    sql = sql.strip().rstrip(";")

    bad = _has_denylisted_keyword(sql)
    if bad:
        raise SQLSafetyError(f"denylisted keyword: {bad}")

    # Must be exactly one statement.
    statements = sqlglot.parse(sql, read="duckdb")
    if len(statements) != 1 or statements[0] is None:
        raise SQLSafetyError(f"expected exactly one SELECT statement; got {len(statements)}")
    tree = statements[0]

    # Statement must be a SELECT (Select / Subquery / WITH ... SELECT).
    if not isinstance(tree, (exp.Select, exp.Subquery)):
        # WITH ... SELECT is a Subquery wrapping a Select — sqlglot models
        # it as `Select` with a `with_` so the isinstance check above
        # already covers it. Anything else is rejected.
        raise SQLSafetyError(f"only SELECT statements are allowed (got {type(tree).__name__})")

    # CTE-introduced names are alias-like — they don't have to be on the
    # allowlist, since the CTE body itself is validated below as part of
    # find_all(exp.Table).
    cte_names = {a.alias for a in tree.find_all(exp.CTE) if a.alias}

    # Tables: every referenced table must be on the allowlist (or be a
    # CTE-introduced name).
    tables = {t.name for t in tree.find_all(exp.Table)}
    bad_tables = tables - allowed_tables - cte_names
    if bad_tables:
        raise SQLSafetyError(f"disallowed table(s): {sorted(bad_tables)}")

    # Columns: every (qualified or unqualified) column reference must be
    # on the allowlist for *some* allowed table. Aggregates (COUNT(*),
    # MAX(...)), aliases, and CTE outputs are ignored — sqlglot already
    # distinguishes those from raw column references.
    allowed_cols: set[str] = set()
    for t in tables:
        allowed_cols |= ALLOWED_COLUMNS.get(t, frozenset())
    referenced_cols = {c.name for c in tree.find_all(exp.Column)}
    # SELECT-list aliases (`COUNT(*) AS n`, `ROUND(...) AS avg`) are
    # legal references in ORDER BY / HAVING / outer SELECT.
    select_aliases: set[str] = set()
    for a in tree.find_all(exp.Alias):
        if a.alias:
            select_aliases.add(a.alias)

    # Allow `*` (Star) — sqlglot models it separately, doesn't appear in
    # find_all(exp.Column). CTE names + SELECT aliases collected above
    # are also acceptable column qualifiers.
    bad_cols = referenced_cols - allowed_cols - cte_names - select_aliases
    if bad_cols:
        raise SQLSafetyError(f"disallowed column(s): {sorted(bad_cols)}")

    return tree


# ── Execution ─────────────────────────────────────────────────────────────


def execute_sql(
    sql: str,
    parquet_path: Path,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_sec: float = DEFAULT_EXEC_SECONDS,
):
    """Execute a *validated* SQL string against the curated parquet.

    Wraps the user SQL in a ``LIMIT`` so we never return more than
    ``max_rows``. ``parquet_path`` is mounted as the ``jobs`` table.

    Returns a pandas DataFrame on success; raises ``SQLExecutionError``
    on DuckDB error or timeout.
    """
    import duckdb

    # Cap rows by wrapping the user SQL.
    sql = sql.strip().rstrip(";")
    capped = f"SELECT * FROM ({sql}) AS _user_query LIMIT {int(max_rows)}"

    con = duckdb.connect()
    try:
        # DuckDB has a hard wall-clock cap via .interrupt() but not a
        # built-in timeout for inline queries — best effort: set a per-
        # connection time limit when supported. Newer versions accept
        # SET statement_timeout; we install a try/except in case the
        # build doesn't support it.
        with contextlib.suppress(Exception):
            con.execute(f"SET statement_timeout = {int(timeout_sec * 1000)}")
        con.execute(
            f"CREATE OR REPLACE TEMPORARY VIEW jobs AS "
            f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        )
        df = con.execute(capped).fetch_df()
    except Exception as exc:
        raise SQLExecutionError(str(exc)) from exc
    finally:
        con.close()
    return df


# ── LLM client abstraction ────────────────────────────────────────────────


@dataclass
class LLMClient(ABC):
    """Pluggable LLM backend. Pure synchronous interface — async lands
    when we need it for batched calls."""

    @abstractmethod
    def generate(self, system: str, user: str, *, max_tokens: int = 512) -> str: ...


@dataclass
class MockLLM(LLMClient):
    """Test-only client. Returns ``response_for(user)`` or a default."""

    response: str = ""
    response_for: dict[str, str] = field(default_factory=dict)

    def generate(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        return self.response_for.get(user, self.response)


@dataclass
class HFInferenceLLM(LLMClient):
    """HF Inference API client — defaults to Qwen2.5-7B-Instruct.

    Requires ``HF_TOKEN`` in the env. Free tier should suffice for the
    NL→SQL traffic volume we expect on the demo Space.
    """

    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    timeout_sec: float = 30.0

    def generate(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_TOKEN")
        client = InferenceClient(model=self.model_id, token=token, timeout=self.timeout_sec)
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


@dataclass
class AnthropicLLM(LLMClient):
    """Anthropic Claude client — preferred when ``ANTHROPIC_API_KEY`` is set."""

    model_id: str = "claude-sonnet-4-6"
    timeout_sec: float = 30.0

    def generate(self, system: str, user: str, *, max_tokens: int = 512) -> str:
        from anthropic import Anthropic

        client = Anthropic(timeout=self.timeout_sec)
        resp = client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate text blocks (skip tool calls / thinking blocks).
        return "".join(b.text for b in resp.content if hasattr(b, "text"))


def default_llm() -> LLMClient:
    """Pick an LLM based on env vars.

    Priority: ANTHROPIC_API_KEY → HF_TOKEN → fail loudly.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM()
    if os.environ.get("HF_TOKEN"):
        return HFInferenceLLM()
    raise RuntimeError("No LLM backend configured. Set ANTHROPIC_API_KEY or HF_TOKEN.")


# ── Prompt + orchestrator ─────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a careful analyst writing DuckDB SQL over a single \
table called `jobs`. You MUST:

1. Output ONLY a single DuckDB SELECT statement — no explanations, no \
markdown fences, no other text.
2. Use only columns listed in the schema. Do NOT invent columns.
3. Use ANSI SQL where possible; DuckDB syntax otherwise (e.g. \
`date_trunc('month', posted_at)`, `list_contains(extracted_skills_v1, 'Python')`).
4. Always cap your result with an explicit `LIMIT` clause; default to \
`LIMIT 100` if the user didn't specify.
5. NEVER write INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/COPY/ATTACH/PRAGMA \
or any other non-SELECT statement.

When the question is ambiguous, prefer the simplest valid SELECT that \
answers a literal interpretation of the question."""


@dataclass
class NL2SQLResult:
    """One full NL→SQL→results round-trip, including any failure detail."""

    question: str
    sql: str | None
    rows: Any | None  # pandas DataFrame
    error: str | None
    n_rows: int = 0


def nl_to_sql(
    question: str,
    parquet_path: Path,
    *,
    llm: LLMClient | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_sec: float = DEFAULT_EXEC_SECONDS,
) -> NL2SQLResult:
    """The full pipeline: question → LLM → validate → execute → result.

    All failures are returned in ``NL2SQLResult.error`` rather than
    raised, so the Gradio tab can render the error message and the
    rejected SQL alongside the schema for debugging.
    """
    if llm is None:
        try:
            llm = default_llm()
        except RuntimeError as exc:
            return NL2SQLResult(
                question=question,
                sql=None,
                rows=None,
                error=str(exc),
            )

    user_prompt = (
        f"Schema:\n{schema_description()}\n\n"
        f"Question: {question}\n\n"
        f"Output: ONE DuckDB SELECT statement, no explanation."
    )
    try:
        raw_sql = llm.generate(SYSTEM_PROMPT, user_prompt, max_tokens=512)
    except Exception as exc:  # noqa: BLE001
        return NL2SQLResult(
            question=question,
            sql=None,
            rows=None,
            error=f"LLM call failed: {exc}",
        )

    sql = _strip_sql_fences(raw_sql).strip()
    try:
        validate_sql(sql)
    except SQLSafetyError as exc:
        return NL2SQLResult(
            question=question,
            sql=sql,
            rows=None,
            error=f"SQL rejected: {exc}",
        )

    try:
        df = execute_sql(
            sql,
            parquet_path,
            max_rows=max_rows,
            timeout_sec=timeout_sec,
        )
    except SQLExecutionError as exc:
        return NL2SQLResult(
            question=question,
            sql=sql,
            rows=None,
            error=f"Execution failed: {exc}",
        )

    return NL2SQLResult(
        question=question,
        sql=sql,
        rows=df,
        error=None,
        n_rows=len(df),
    )


_FENCE_RE = re.compile(r"```(?:sql|duckdb)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_sql_fences(text: str) -> str:
    """Remove ```sql ... ``` fences if the LLM emitted them."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


# ── JSON serialization for the Gradio tab ─────────────────────────────────


def serialize_result(result: NL2SQLResult) -> dict[str, Any]:
    """Lossy → JSON-friendly view of a NL2SQLResult for logging / UI."""
    rows_preview: list[dict[str, Any]] | None = None
    if result.rows is not None:
        rows_preview = result.rows.head(20).to_dict(orient="records")
    return {
        "question": result.question,
        "sql": result.sql,
        "n_rows": result.n_rows,
        "error": result.error,
        "rows_preview": rows_preview,
    }


__all__ = [
    "ALLOWED_COLUMNS",
    "ALLOWED_TABLES",
    "DEFAULT_EXEC_SECONDS",
    "DEFAULT_MAX_ROWS",
    "AnthropicLLM",
    "HFInferenceLLM",
    "LLMClient",
    "MockLLM",
    "NL2SQLResult",
    "SQLExecutionError",
    "SQLSafetyError",
    "default_llm",
    "execute_sql",
    "nl_to_sql",
    "schema_description",
    "serialize_result",
    "validate_sql",
]


def main() -> int:
    """Manual CLI for debugging — prints the SQL + results for one question."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question", help="Natural-language question")
    p.add_argument(
        "--curated-path", default=None, help="Defaults to data/curated_enriched/jobs.parquet"
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    cp = Path(args.curated_path or "data/curated_enriched/jobs.parquet")
    if not cp.exists():
        cp = Path("data/curated/jobs.parquet")
    result = nl_to_sql(args.question, cp)
    print(json.dumps(serialize_result(result), indent=2, default=str))
    return 0 if result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
