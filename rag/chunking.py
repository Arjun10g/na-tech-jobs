"""Parent-child recursive chunking for the RAG retrieval index.

Per CLAUDE.md §8 every job's ``description_md`` is split with two parallel
chunkings:

- **Child chunks** (~256 tokens, 32 overlap) — used for first-pass dense /
  sparse retrieval. Smaller chunks = higher precision: a single section
  about salary doesn't drag a "remote-friendly" mention along with it
  when the user's query only matches one of them.
- **Parent chunks** (~1024 tokens, no overlap) — returned to the LLM /
  shown in UI. After we retrieve the top-K children we hydrate to their
  parent so the model sees the full surrounding context.

Each chunk gets a payload with the full set of fields CLAUDE.md §8
specifies for Qdrant filtering at query time
(``predicted_salary_usd_v1``, ``seniority_label_v1``, ``country``, ...).

Token estimates use a rough 4-chars-per-token heuristic, not a real
tokenizer. The size targets are for retrieval quality, not exact
budgeting; bge-m3 truncates anything past 8k anyway.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# CLAUDE.md §8: hierarchical separators preserve markdown structure.
DEFAULT_SEPARATORS = ("\n## ", "\n### ", "\n\n", "\n", ". ", " ")
CHARS_PER_TOKEN = 4

DEFAULT_CHILD_TOKENS = 256
DEFAULT_CHILD_OVERLAP_TOKENS = 32
DEFAULT_PARENT_TOKENS = 1024
DEFAULT_PARENT_OVERLAP_TOKENS = 0

# Payload columns that downstream Qdrant filters care about — kept stable so
# the indexer doesn't have to re-discover them. Anything not on this list
# gets dropped at indexing time to keep the payload small.
PAYLOAD_FIELDS: tuple[str, ...] = (
    "id",
    "company_slug",
    "company_name",
    "title",
    "url",
    "country",
    "region",
    "city",
    "remote_policy",
    "source",
    "posted_at",
    "salary_min_usd_yearly",
    "salary_max_usd_yearly",
    "salary_disclosed",
    "seniority_extracted",
    "role_family_extracted",
    # Phase 4 enrichment — present only on curated_enriched/jobs.parquet.
    "seniority_label_v1",
    "seniority_confidence_v1",
    "role_family_v1",
    "role_family_confidence_v1",
    "predicted_salary_usd_v1",
    "extracted_skills_v1",
    "prediction_model_version",
)


@dataclass
class ParentChunk:
    """A larger context window — what the LLM sees after hydration."""

    parent_chunk_id: str  # f"{job_id}::p{i}"
    job_id: str
    chunk_index: int
    text: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildChunk:
    """A smaller window — what we actually search over."""

    child_chunk_id: str  # f"{job_id}::c{i}"
    job_id: str
    chunk_index: int
    parent_chunk_id: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)


def _build_splitter(target_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=list(DEFAULT_SEPARATORS),
        chunk_size=target_tokens * CHARS_PER_TOKEN,
        chunk_overlap=overlap_tokens * CHARS_PER_TOKEN,
        length_function=len,
        is_separator_regex=False,
    )


def _select_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Extract only the payload fields that survive into the index."""
    payload: dict[str, Any] = {}
    for k in PAYLOAD_FIELDS:
        if k in row:
            v = row[k]
            # Coerce numpy/pandas scalars to plain Python so the payload
            # serializes cleanly in Qdrant.
            if hasattr(v, "item"):
                with contextlib.suppress(AttributeError, ValueError):
                    v = v.item()
            elif hasattr(v, "tolist"):
                with contextlib.suppress(AttributeError, ValueError):
                    v = v.tolist()
            payload[k] = v
    return payload


def chunk_job(
    row: dict[str, Any],
    *,
    child_tokens: int = DEFAULT_CHILD_TOKENS,
    child_overlap_tokens: int = DEFAULT_CHILD_OVERLAP_TOKENS,
    parent_tokens: int = DEFAULT_PARENT_TOKENS,
    parent_overlap_tokens: int = DEFAULT_PARENT_OVERLAP_TOKENS,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Split one job's description into parent and child chunks.

    The title is prepended to the description so retrieval matches on
    title-only queries (e.g. "ML Engineer") which would otherwise miss the
    body. Empty descriptions are tolerated — we still index a single chunk
    holding just the title.
    """
    job_id = row["id"]
    title = (row.get("title") or "").strip()
    body = (row.get("description_md") or "").strip()
    text = f"# {title}\n\n{body}".strip() if title else body
    if not text:
        return [], []

    payload = _select_payload(row)

    parent_splitter = _build_splitter(parent_tokens, parent_overlap_tokens)
    parent_texts = parent_splitter.split_text(text) or [text]
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for p_idx, p_text in enumerate(parent_texts):
        parent_id = f"{job_id}::p{p_idx}"
        parents.append(
            ParentChunk(
                parent_chunk_id=parent_id,
                job_id=job_id,
                chunk_index=p_idx,
                text=p_text,
                payload=payload,
            )
        )

        child_splitter = _build_splitter(child_tokens, child_overlap_tokens)
        # Children for *this* parent only. The parent is already small enough
        # that a single child often suffices — but for ~1024-token parents,
        # we typically split into 4-5 child windows.
        child_texts = child_splitter.split_text(p_text) or [p_text]
        for c_text in child_texts:
            children.append(
                ChildChunk(
                    child_chunk_id=f"{job_id}::c{len(children):04d}",
                    job_id=job_id,
                    chunk_index=len(children),
                    parent_chunk_id=parent_id,
                    text=c_text,
                    payload=payload,
                )
            )

    return parents, children


def chunk_jobs(
    rows: list[dict[str, Any]],
    **kwargs,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Chunk a batch of jobs. Pure function over per-row dicts —
    callers convert from a DataFrame with ``df.to_dict(orient="records")``.
    """
    all_parents: list[ParentChunk] = []
    all_children: list[ChildChunk] = []
    for row in rows:
        parents, children = chunk_job(row, **kwargs)
        all_parents.extend(parents)
        all_children.extend(children)
    return all_parents, all_children
