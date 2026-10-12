"""Retrieval tests: real Postgres + pgvector + HNSW, with the word-overlap fake embedder."""

import asyncio
import os
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.fakes import FakeEmbedder

GOVERNANCE = (
    b"# Governance\n\nThe governance function assigns accountability and oversight "
    b"for AI risk. Leadership approves policies."
)
MEASUREMENT = (
    b"# Measurement\n\nMeasurement uses metrics and benchmarks to evaluate model "
    b"accuracy, bias, and robustness."
)
INCIDENTS = (
    b"# Incidents\n\nIncident response plans describe how to report and recover from AI failures."
)


def _collection(api: TestClient, name: str) -> str:
    response = api.post("/v1/collections", json={"name": name})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _upload(api: TestClient, collection_id: str, filename: str, data: bytes) -> str:
    response = api.post(
        f"/v1/collections/{collection_id}/documents", files={"file": (filename, data)}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _search(api: TestClient, collection_id: str, query: str, **extra: object) -> dict:  # type: ignore[type-arg]
    response = api.post(f"/v1/collections/{collection_id}/search", json={"query": query, **extra})
    assert response.status_code == 200, response.text
    result: dict = response.json()  # type: ignore[type-arg]
    return result


def _seeded(api: TestClient, name: str = "nist") -> str:
    collection_id = _collection(api, name)
    _upload(api, collection_id, "governance.md", GOVERNANCE)
    _upload(api, collection_id, "measurement.md", MEASUREMENT)
    _upload(api, collection_id, "incidents.md", INCIDENTS)
    return collection_id


def test_most_relevant_chunk_ranks_first(api: TestClient) -> None:
    collection_id = _seeded(api)

    body = _search(api, collection_id, "Who has accountability and oversight for AI risk?")

    top = body["results"][0]
    assert top["filename"] == "governance.md"
    assert top["heading"] == "Governance"
    assert "accountability" in top["content"]


def test_results_are_sorted_by_score_and_scores_are_similarities(api: TestClient) -> None:
    collection_id = _seeded(api)

    results = _search(api, collection_id, "metrics to evaluate model bias")["results"]

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert results[0]["filename"] == "measurement.md"


def test_top_k_limits_results(api: TestClient) -> None:
    collection_id = _seeded(api)

    assert len(_search(api, collection_id, "AI", top_k=2)["results"]) == 2
    assert len(_search(api, collection_id, "AI", top_k=10)["results"]) == 3  # only 3 chunks exist


def test_results_include_source_metadata(api: TestClient) -> None:
    collection_id = _collection(api, "pdf")
    pdf = (Path(__file__).parent.parent / "fixtures" / "sample.pdf").read_bytes()
    document_id = _upload(api, collection_id, "sample.pdf", pdf)

    result = _search(api, collection_id, "MEASURE function quantitative tools")["results"][0]

    assert result["document_id"] == document_id
    assert (result["page_start"], result["page_end"]) == (1, 2)
    assert uuid.UUID(result["chunk_id"])


def test_search_never_crosses_collections(api: TestClient) -> None:
    """Tenant isolation: identical content in another collection must not leak in."""
    mine = _collection(api, "mine")
    theirs = _collection(api, "theirs")
    _upload(api, mine, "measurement.md", MEASUREMENT)
    _upload(api, theirs, "governance.md", GOVERNANCE)

    results = _search(api, mine, "accountability and oversight", top_k=10)["results"]

    assert [r["filename"] for r in results] == ["measurement.md"]


def test_small_collection_still_gets_results_among_many_others(api: TestClient) -> None:
    """Regression test for HNSW filtering: the index first finds neighbors across ALL
    collections, then the WHERE clause filters. Iterative scan keeps searching so a
    small collection isn't starved by a big one."""
    big = _collection(api, "big")
    for n in range(30):
        _upload(api, big, f"gov-{n}.md", GOVERNANCE + f" Copy {n}.".encode())
    small = _collection(api, "small")
    _upload(api, small, "incidents.md", INCIDENTS)

    results = _search(api, small, "governance accountability oversight", top_k=5)["results"]

    assert [r["filename"] for r in results] == ["incidents.md"]


def test_empty_collection_returns_no_results(api: TestClient) -> None:
    collection_id = _collection(api, "empty")

    assert _search(api, collection_id, "anything")["results"] == []


def test_only_ready_documents_are_searched(api: TestClient) -> None:
    """Chunks of a document that is processing or failed must never be used to answer."""
    collection_id = _seeded(api)
    extra_id = _upload(api, collection_id, "extra.md", b"# Extra\n\nGovernance audits.")

    async def mark_failed() -> None:
        engine = create_async_engine(os.environ["DATABASE_URL"])
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE documents SET status = 'failed' WHERE id = :id"),
                {"id": extra_id},
            )
        await engine.dispose()

    asyncio.run(mark_failed())
    body = _search(api, collection_id, "Governance audits", top_k=10)

    assert body["results"], "ready documents are still searched"
    assert extra_id not in {r["document_id"] for r in body["results"]}


def test_query_is_embedded_once_with_the_document_model(
    api: TestClient, fake_embedder: FakeEmbedder
) -> None:
    collection_id = _seeded(api)
    calls_before = len(fake_embedder.calls)

    _search(api, collection_id, "incident response")

    assert fake_embedder.calls[calls_before:] == [["incident response"]]


def test_missing_collection_is_404(api: TestClient) -> None:
    response = api.post(f"/v1/collections/{uuid.uuid4()}/search", json={"query": "x"})

    assert response.status_code == 404


def test_invalid_requests_are_422(api: TestClient) -> None:
    collection_id = _collection(api, "v")
    url = f"/v1/collections/{collection_id}/search"

    assert api.post(url, json={"query": ""}).status_code == 422
    assert api.post(url, json={"query": "x" * 1001}).status_code == 422
    assert api.post(url, json={"query": "x", "top_k": 0}).status_code == 422
    assert api.post(url, json={"query": "x", "top_k": 21}).status_code == 422


def test_search_needs_no_admin_key(api: TestClient) -> None:
    collection_id = _seeded(api)

    response = api.post(
        f"/v1/collections/{collection_id}/search",
        json={"query": "risk"},
        headers={"X-API-Key": ""},
    )

    assert response.status_code == 200


def test_embedding_failure_is_502(api: TestClient, fake_embedder: FakeEmbedder) -> None:
    collection_id = _seeded(api)
    fake_embedder.fail = True

    response = api.post(f"/v1/collections/{collection_id}/search", json={"query": "risk"})

    assert response.status_code == 502


def test_search_query_can_use_the_hnsw_index() -> None:
    """The query shape (ORDER BY distance LIMIT k) must be index-compatible.

    With tiny test tables Postgres rightly prefers an exact scan + sort (cheaper
    than an approximate graph walk for a few rows). We disable those plans to ask:
    COULD it use the index once the data is large? If someone rewrites the query
    in a way the index can't serve, this fails.
    """

    async def plan() -> str:
        engine = create_async_engine(os.environ["DATABASE_URL"])
        async with engine.connect() as conn:
            # Rule out the plans Postgres prefers for tiny tables (full scan, or
            # fetch-the-collection-then-sort), leaving only the index path if valid.
            await conn.execute(text("SET enable_seqscan = off"))
            await conn.execute(text("SET enable_sort = off"))
            vector = "[" + ",".join(["0.1"] * 1536) + "]"
            rows = await conn.execute(
                text(
                    "EXPLAIN SELECT chunks.id FROM chunks "
                    "JOIN documents ON documents.id = chunks.document_id "
                    "WHERE documents.collection_id = :cid "
                    "ORDER BY chunks.embedding <=> CAST(:v AS vector) LIMIT 5"
                ),
                {"cid": str(uuid.uuid4()), "v": vector},
            )
            output = "\n".join(row[0] for row in rows)
        await engine.dispose()
        return output

    assert "ix_chunks_embedding_hnsw" in asyncio.run(plan())


def test_search_rate_limit_is_per_client(api: TestClient) -> None:
    from grounded.api.rate_limit import SlidingWindowRateLimiter

    collection_id = _seeded(api)
    api.app.state.rate_limiters["search"] = SlidingWindowRateLimiter(limit=1)  # type: ignore[attr-defined]
    url = f"/v1/collections/{collection_id}/search"

    assert api.post(url, json={"query": "risk"}).status_code == 200
    assert api.post(url, json={"query": "risk"}).status_code == 429
