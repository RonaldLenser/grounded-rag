"""End-to-end question answering: real Postgres, fake embeddings, fake LLM."""

import asyncio
import os
import uuid

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from grounded.db.models import QueryLog
from grounded.generation.prompts import INSUFFICIENT_CONTEXT_MESSAGE
from tests.fakes import FakeLLM

GOVERNANCE = (
    b"# Governance\n\nThe governance function assigns accountability and oversight for AI "
    b"risk. Leadership approves policies."
)
MEASUREMENT = (
    b"# Measurement\n\nMeasurement uses metrics and benchmarks to evaluate model accuracy, "
    b"bias, and robustness."
)
QUESTION = "Who has accountability and oversight for AI risk?"


def _seeded(api: TestClient) -> str:
    collection_id = str(api.post("/v1/collections", json={"name": "qa"}).json()["id"])
    for name, data in (("governance.md", GOVERNANCE), ("measurement.md", MEASUREMENT)):
        response = api.post(
            f"/v1/collections/{collection_id}/documents", files={"file": (name, data)}
        )
        assert response.status_code == 201, response.text
    return collection_id


def _ask(api: TestClient, collection_id: str, question: str = QUESTION) -> dict:  # type: ignore[type-arg]
    response = api.post(f"/v1/collections/{collection_id}/query", json={"question": question})
    assert response.status_code == 200, response.text
    body: dict = response.json()  # type: ignore[type-arg]
    return body


def _query_logs() -> list[QueryLog]:
    async def fetch() -> list[QueryLog]:
        engine = create_async_engine(os.environ["DATABASE_URL"])
        async with AsyncSession(engine) as session:
            rows = list((await session.scalars(select(QueryLog))).all())
        await engine.dispose()
        return rows

    return asyncio.run(fetch())


def test_grounded_answer_with_verified_citation(api: TestClient) -> None:
    collection_id = _seeded(api)

    body = _ask(api, collection_id)

    assert body["status"] == "answered"
    assert "[S1]" in body["answer"]
    [citation] = body["citations"]
    assert citation["source_id"] == "S1"
    assert citation["filename"] == "governance.md"
    assert citation["heading"] == "Governance"
    assert citation["quote"] in GOVERNANCE.decode()
    assert body["verification"] == {
        "citations_verified": 1,
        "citations_rejected": 0,
        "downgraded_from": None,
    }
    assert body["usage"]["llm_called"] is True
    assert body["usage"]["model"] == "fake-llm"


def test_prompt_sends_sources_and_question_to_the_llm(api: TestClient, fake_llm: FakeLLM) -> None:
    collection_id = _seeded(api)

    _ask(api, collection_id)

    [call] = fake_llm.calls
    assert call["schema"] == "grounded_answer"
    assert '<source id="S1" file="governance.md">' in call["user"]
    assert f"<question>{QUESTION}</question>" in call["user"]
    assert "untrusted document text" in call["system"]


def test_irrelevant_question_is_refused_without_calling_the_llm(
    api: TestClient, fake_llm: FakeLLM
) -> None:
    collection_id = _seeded(api)

    body = _ask(api, collection_id, "best chocolate chip cookie recipe")

    assert body["status"] == "insufficient_context"
    assert body["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert body["citations"] == []
    assert body["usage"]["llm_called"] is False
    assert fake_llm.calls == [], "the relevance gate must stop the LLM call (cost + safety)"


def test_empty_collection_is_refused_without_calling_the_llm(
    api: TestClient, fake_llm: FakeLLM
) -> None:
    collection_id = str(api.post("/v1/collections", json={"name": "empty"}).json()["id"])

    body = _ask(api, collection_id)

    assert body["status"] == "insufficient_context"
    assert fake_llm.calls == []


def test_hallucinated_citation_downgrades_to_insufficient(
    api: TestClient, fake_llm: FakeLLM
) -> None:
    collection_id = _seeded(api)
    fake_llm.respond_with = {
        "status": "answered",
        "answer": "The CEO personally signs off on every model [S1].",
        "citations": [{"source_id": "S1", "quote": "The CEO personally signs off on every model"}],
    }

    body = _ask(api, collection_id)

    assert body["status"] == "insufficient_context"
    assert body["answer"] == INSUFFICIENT_CONTEXT_MESSAGE, "unverifiable answers are never shown"
    assert body["citations"] == []
    assert body["verification"]["downgraded_from"] == "answered"
    assert body["verification"]["citations_rejected"] == 1


def test_partially_invented_citations_downgrade_to_partial(
    api: TestClient, fake_llm: FakeLLM
) -> None:
    collection_id = _seeded(api)
    fake_llm.respond_with = {
        "status": "answered",
        "answer": "Accountability is assigned [S1]. Budgets are unlimited [S9].",
        "citations": [
            {"source_id": "S1", "quote": "assigns accountability and oversight for AI risk"},
            {"source_id": "S9", "quote": "budgets are unlimited for all teams"},
        ],
    }

    body = _ask(api, collection_id)

    assert body["status"] == "partial"
    assert body["answer"] == "Accountability is assigned [S1]. Budgets are unlimited."
    assert [c["source_id"] for c in body["citations"]] == ["S1"]


def test_model_refusal_passes_through(api: TestClient, fake_llm: FakeLLM) -> None:
    collection_id = _seeded(api)
    fake_llm.respond_with = {
        "status": "insufficient_context",
        "answer": "The documents don't say who audits the models.",
        "citations": [],
    }

    body = _ask(api, collection_id)

    assert body["status"] == "insufficient_context"
    assert body["answer"] == "The documents don't say who audits the models."
    assert body["usage"]["llm_called"] is True


def test_malformed_llm_output_is_502(api: TestClient, fake_llm: FakeLLM) -> None:
    collection_id = _seeded(api)
    fake_llm.respond_with = {"status": "definitely", "answer": 42}

    response = api.post(f"/v1/collections/{collection_id}/query", json={"question": QUESTION})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_provider_error"


def test_llm_outage_is_502(api: TestClient, fake_llm: FakeLLM) -> None:
    collection_id = _seeded(api)
    fake_llm.fail = True

    response = api.post(f"/v1/collections/{collection_id}/query", json={"question": QUESTION})

    assert response.status_code == 502


def test_every_query_is_logged(api: TestClient) -> None:
    collection_id = _seeded(api)

    answered = _ask(api, collection_id)
    refused = _ask(api, collection_id, "best chocolate chip cookie recipe")

    logs = {str(log.id): log for log in _query_logs()}
    assert logs[answered["query_id"]].status == "answered"
    assert logs[answered["query_id"]].model == "fake-llm"
    assert logs[answered["query_id"]].input_tokens == 100
    assert len(logs[answered["query_id"]].retrieved_chunk_ids) == 2
    assert logs[refused["query_id"]].status == "insufficient_context"
    assert logs[refused["query_id"]].model is None


def test_missing_collection_is_404(api: TestClient) -> None:
    response = api.post(f"/v1/collections/{uuid.uuid4()}/query", json={"question": "x"})

    assert response.status_code == 404


def test_question_validation(api: TestClient) -> None:
    collection_id = _seeded(api)
    url = f"/v1/collections/{collection_id}/query"

    assert api.post(url, json={"question": ""}).status_code == 422
    assert api.post(url, json={"question": "x" * 1001}).status_code == 422
