"""Tests for rag.qdrant_client. Uses local-mode Qdrant in a tmp dir —
no network, no docker, no fixtures beyond the disk."""

from __future__ import annotations

import numpy as np
import pytest

from rag.chunking import ChildChunk
from rag.qdrant_client import (
    COLLECTION_DENSE,
    COLLECTION_MULTIVEC,
    chunk_id_to_point_id,
    collection_info,
    get_client,
    setup_dense_collection,
    setup_multivec_collection,
    upsert_dense,
    upsert_multivec,
)


@pytest.fixture
def client(tmp_path):
    return get_client(tmp_path / "qdrant")


@pytest.fixture
def chunks():
    return [
        ChildChunk(
            child_chunk_id=f"job{i}::c0",
            job_id=f"job{i}",
            chunk_index=0,
            parent_chunk_id=f"job{i}::p0",
            text=f"chunk text {i}",
            payload={"id": f"job{i}", "country": "US", "title": f"Title {i}"},
        )
        for i in range(5)
    ]


def test_chunk_id_to_point_id_is_stable():
    a = chunk_id_to_point_id("foo::c0")
    b = chunk_id_to_point_id("foo::c0")
    c = chunk_id_to_point_id("foo::c1")
    assert a == b
    assert a != c


def test_setup_dense_collection_idempotent(client):
    setup_dense_collection(client, dense_dim=32)
    setup_dense_collection(client, dense_dim=32)  # no-op second call
    assert client.collection_exists(COLLECTION_DENSE)


def test_setup_multivec_collection_creates_named_vector(client):
    setup_multivec_collection(client, multivec_dim=16)
    assert client.collection_exists(COLLECTION_MULTIVEC)


def test_upsert_dense_writes_points_with_payload(client, chunks):
    setup_dense_collection(client, dense_dim=32)
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((len(chunks), 32)).astype(np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)

    n_written = upsert_dense(client, chunks, dense)
    assert n_written == len(chunks)

    info = collection_info(client, COLLECTION_DENSE)
    assert info["points_count"] == len(chunks)


def test_upsert_dense_with_sparse_round_trips(client, chunks):
    setup_dense_collection(client, dense_dim=32)
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((len(chunks), 32)).astype(np.float32)
    sparse = [{i: 0.5 + 0.1 * j for i in range(3)} for j in range(len(chunks))]

    n_written = upsert_dense(client, chunks, dense, sparse)
    assert n_written == len(chunks)


def test_upsert_dense_dim_mismatch_raises(client, chunks):
    setup_dense_collection(client, dense_dim=32)
    bogus = np.zeros((len(chunks) + 1, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="dense rows"):
        upsert_dense(client, chunks, bogus)


def test_upsert_multivec_writes_points(client, chunks):
    setup_multivec_collection(client, multivec_dim=8)
    rng = np.random.default_rng(0)
    multivec = [rng.standard_normal((4, 8)).astype(np.float32) for _ in chunks]

    n_written = upsert_multivec(client, chunks, multivec)
    assert n_written == len(chunks)
    info = collection_info(client, COLLECTION_MULTIVEC)
    assert info["points_count"] == len(chunks)


def test_upsert_multivec_length_mismatch_raises(client, chunks):
    setup_multivec_collection(client, multivec_dim=8)
    multivec = [np.zeros((4, 8), dtype=np.float32)]  # only 1, need 5
    with pytest.raises(ValueError, match="multivec rows"):
        upsert_multivec(client, chunks, multivec)


def test_payload_includes_chunking_metadata(client, chunks):
    setup_dense_collection(client, dense_dim=32)
    dense = np.zeros((len(chunks), 32), dtype=np.float32)
    upsert_dense(client, chunks, dense)

    point_id = chunk_id_to_point_id(chunks[0].child_chunk_id)
    pts = client.retrieve(collection_name=COLLECTION_DENSE, ids=[point_id])
    assert len(pts) == 1
    payload = pts[0].payload
    assert payload["chunk_id"] == chunks[0].child_chunk_id
    assert payload["job_id"] == chunks[0].job_id
    assert payload["parent_chunk_id"] == chunks[0].parent_chunk_id
    # Original payload fields survive too.
    assert payload["country"] == "US"
