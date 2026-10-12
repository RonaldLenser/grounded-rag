"""Hybrid retrieval against a real Postgres: keyword weighting, fusion, isolation.

The embedder here returns the SAME vector for every text, so vector search can't
tell chunks apart. Whatever ranks first must have been put there by keyword search.
"""

import asyncio
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from grounded.api.deps import get_embedder
from grounded.db.models import EMBEDDING_DIM
from tests.fakes import FakeLLM

COMMON = "AI risk management helps organizations govern and measure AI risk."
DOCS = {
    "common-1.md": f"# One\n\n{COMMON} Leadership approves risk policies.",
    "common-2.md": f"# Two\n\n{COMMON} Teams review risk every quarter.",
    "common-3.md": f"# Three\n\n{COMMON} Risk owners are assigned for each system.",
    "residual.md": "# Residual\n\nNegative residual risks are documented for acquirers.",
}


class ConstantEmbedder:
    """Every text gets the same vector: vector search ranks are uninformative."""

    model = "constant"
    dimension = EMBEDDING_DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]


@pytest.fixture
def hybrid_api(api: TestClient) -> TestClient:
    api.app.dependency_overrides[get_embedder] = lambda: ConstantEmbedder()  # type: ignore[attr-defined]
    api.app.state.settings.retrieval_strategy = "hybrid"  # type: ignore[attr-defined]
    return api


def _seed(api: TestClient, name: str = "hybrid", docs: dict[str, str] = DOCS) -> str:
    collection_id = str(api.post("/v1/collections", json={"name": name}).json()["id"])
    for filename, content in docs.items():
        response = api.post(
            f"/v1/collections/{collection_id}/documents",
            files={"file": (filename, content.encode())},
        )
        assert response.status_code == 201, response.text
    return collection_id


def _search(api: TestClient, collection_id: str, query: str, top_k: int = 4) -> dict:  # type: ignore[type-arg]
    response = api.post(
        f"/v1/collections/{collection_id}/search", json={"query": query, "top_k": top_k}
    )
    assert response.status_code == 200, response.text
    body: dict = response.json()  # type: ignore[type-arg]
    return body


def test_rare_words_outweigh_words_on_every_page(hybrid_api: TestClient) -> None:
    """'risk' is in every chunk (weight ~0); 'residual' is in one (high weight)."""
    collection_id = _seed(hybrid_api)

    body = _search(hybrid_api, collection_id, "What is residual risk?")

    assert body["strategy"] == "hybrid"
    top = body["results"][0]
    assert top["filename"] == "residual.md"
    assert top["keyword_rank"] == 1
    assert top["vector_rank"] is not None
    assert top["fused_score"] > body["results"][1]["fused_score"]


class AwayFromResidualEmbedder:
    """The residual chunk points one way, everything else (and the question) another.

    Vector search therefore ranks it LAST, so with 2 candidates it is never one of
    them: only keyword search can bring it back.
    """

    model = "away"
    dimension = EMBEDDING_DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def vector(t: str) -> list[float]:
            axis = 1 if "acquirers" in t else 0
            return [1.0 if i == axis else 0.0 for i in range(EMBEDDING_DIM)]

        return [vector(t) for t in texts]


def test_keyword_only_chunks_still_get_their_cosine_score(hybrid_api: TestClient) -> None:
    """A chunk outside the vector candidates needs its real similarity for the gate."""
    hybrid_api.app.dependency_overrides[get_embedder] = lambda: AwayFromResidualEmbedder()  # type: ignore[attr-defined]
    hybrid_api.app.state.settings.hybrid_candidates = 2  # type: ignore[attr-defined]
    collection_id = _seed(hybrid_api)

    results = _search(hybrid_api, collection_id, "residual", top_k=2)["results"]

    [keyword_only] = [r for r in results if r["vector_rank"] is None]
    assert keyword_only["filename"] == "residual.md"
    assert keyword_only["keyword_rank"] == 1
    assert keyword_only["score"] == pytest.approx(0.0), "its true cosine, not a placeholder"


def test_question_of_only_common_words_falls_back_to_vector_order(hybrid_api: TestClient) -> None:
    collection_id = _seed(hybrid_api)

    for query in ("the and of", "risk"):  # stopwords only / a word in every chunk
        results = _search(hybrid_api, collection_id, query)["results"]
        assert len(results) == 4
        assert all(r["keyword_rank"] is None for r in results)


def test_keyword_search_never_crosses_collections(hybrid_api: TestClient) -> None:
    mine = _seed(hybrid_api, "mine", {"a.md": "# A\n\nGovernance is cross-cutting."})
    _seed(hybrid_api, "theirs", {"b.md": "# B\n\nNegative residual risks are documented."})

    results = _search(hybrid_api, mine, "residual risks documented")["results"]

    assert {r["filename"] for r in results} == {"a.md"}


def test_keyword_search_skips_documents_that_are_not_ready(hybrid_api: TestClient) -> None:
    collection_id = _seed(hybrid_api)

    async def mark_failed() -> None:
        engine = create_async_engine(os.environ["DATABASE_URL"])
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE documents SET status = 'failed' WHERE filename = 'residual.md'")
            )
        await engine.dispose()

    asyncio.run(mark_failed())
    results = _search(hybrid_api, collection_id, "residual", top_k=10)["results"]

    assert "residual.md" not in {r["filename"] for r in results}


def test_query_reports_strategy_and_answers_from_the_keyword_match(
    hybrid_api: TestClient, fake_llm: FakeLLM
) -> None:
    collection_id = _seed(hybrid_api)

    response = hybrid_api.post(
        f"/v1/collections/{collection_id}/query", json={"question": "What is residual risk?"}
    )

    body = response.json()
    assert body["retrieval"]["strategy"] == "hybrid"
    assert body["status"] == "answered"
    assert '<source id="S1" file="residual.md">' in fake_llm.calls[0]["user"]


def test_vector_mode_reports_no_ranks(api: TestClient) -> None:
    collection_id = _seed(api, "vec")

    body = _search(api, collection_id, "residual risk")

    assert body["strategy"] == "vector"
    assert all(r["keyword_rank"] is None and r["fused_score"] is None for r in body["results"])
