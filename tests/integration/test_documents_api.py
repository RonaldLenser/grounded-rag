"""End-to-end ingestion tests: HTTP request → real Postgres, with fake embeddings."""

import asyncio
import os
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from grounded.db.models import Chunk, Document
from tests.fakes import FakeEmbedder

FIXTURES = Path(__file__).parent.parent / "fixtures"
MARKDOWN = (
    b"# Policy\n\nAll AI systems must be inventoried. Owners are assigned.\n\n"
    b"## Review\n\nReview yearly."
)


def _db_counts() -> tuple[int, int]:
    async def count() -> tuple[int, int]:
        engine = create_async_engine(os.environ["DATABASE_URL"])
        async with AsyncSession(engine) as session:
            docs = await session.scalar(select(func.count(Document.id))) or 0
            chunks = await session.scalar(select(func.count(Chunk.id))) or 0
        await engine.dispose()
        return docs, chunks

    return asyncio.run(count())


def _create_collection(api: TestClient, name: str = "test-docs") -> str:
    response = api.post("/v1/collections", json={"name": name})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _upload(api: TestClient, collection_id: str, filename: str, data: bytes) -> dict:  # type: ignore[type-arg]
    response = api.post(
        f"/v1/collections/{collection_id}/documents", files={"file": (filename, data)}
    )
    return {"status": response.status_code, "body": response.json()}


# ----- Collections -----
def test_create_and_list_collections(api: TestClient) -> None:
    collection_id = _create_collection(api, "alpha")

    listed = api.get("/v1/collections").json()

    assert [c["id"] for c in listed] == [collection_id]
    assert listed[0]["name"] == "alpha"


def test_duplicate_collection_name_is_409(api: TestClient) -> None:
    _create_collection(api, "alpha")

    response = api.post("/v1/collections", json={"name": "alpha"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


def test_invalid_collection_name_uses_standard_error_shape(api: TestClient) -> None:
    response = api.post("/v1/collections", json={"name": "../etc/passwd"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["request_id"] == response.headers["X-Request-ID"]


# ----- Auth -----
def test_writes_require_the_admin_key(api: TestClient) -> None:
    response = api.post("/v1/collections", json={"name": "x"}, headers={"X-API-Key": "wrong-key"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_reads_do_not_require_a_key(api: TestClient) -> None:
    response = api.get("/v1/collections", headers={"X-API-Key": ""})

    assert response.status_code == 200


# ----- Upload: happy paths -----
def test_upload_pdf_stores_document_and_chunks(
    api: TestClient, fake_embedder: FakeEmbedder
) -> None:
    collection_id = _create_collection(api)

    result = _upload(api, collection_id, "sample.pdf", (FIXTURES / "sample.pdf").read_bytes())

    assert result["status"] == 201
    body = result["body"]
    assert body["filename"] == "sample.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["page_count"] == 2
    assert body["status"] == "ready"
    assert body["chunk_count"] >= 1
    assert _db_counts() == (1, body["chunk_count"])
    # Every stored chunk was embedded, in one batched call.
    assert len(fake_embedder.calls) == 1
    assert len(fake_embedder.calls[0]) == body["chunk_count"]


def test_upload_markdown_keeps_headings(api: TestClient) -> None:
    collection_id = _create_collection(api)

    result = _upload(api, collection_id, "policy.md", MARKDOWN)

    assert result["status"] == 201
    assert result["body"]["chunk_count"] == 2  # one chunk per section
    assert result["body"]["page_count"] is None


def test_list_get_and_delete_document(api: TestClient) -> None:
    collection_id = _create_collection(api)
    document_id = _upload(api, collection_id, "policy.md", MARKDOWN)["body"]["id"]

    listed = api.get(f"/v1/collections/{collection_id}/documents").json()
    fetched = api.get(f"/v1/documents/{document_id}").json()
    deleted = api.delete(f"/v1/documents/{document_id}")

    assert [d["id"] for d in listed] == [document_id]
    assert listed[0]["chunk_count"] == 2
    assert fetched["chunk_count"] == 2
    assert deleted.status_code == 204
    assert api.get(f"/v1/documents/{document_id}").status_code == 404
    assert _db_counts() == (0, 0), "deleting a document must delete its chunks"


# ----- Upload: rejections -----
def test_same_file_twice_is_409_even_if_renamed(api: TestClient) -> None:
    collection_id = _create_collection(api)
    first = _upload(api, collection_id, "policy.md", MARKDOWN)

    second = _upload(api, collection_id, "renamed-copy.md", MARKDOWN)

    assert second["status"] == 409
    assert second["body"]["error"]["code"] == "duplicate_document"
    assert second["body"]["error"]["existing_document_id"] == first["body"]["id"]


def test_same_file_in_different_collections_is_allowed(api: TestClient) -> None:
    first = _upload(api, _create_collection(api, "one"), "policy.md", MARKDOWN)
    second = _upload(api, _create_collection(api, "two"), "policy.md", MARKDOWN)

    assert (first["status"], second["status"]) == (201, 201)


@pytest.mark.parametrize(
    ("filename", "data", "status", "code"),
    [
        ("report.docx", b"PK\x03\x04", 415, "unsupported_document"),
        ("fake.pdf", b"not really a pdf", 415, "unsupported_document"),
        ("latin1.txt", "caf\xe9".encode("latin-1"), 422, "unprocessable_document"),
        ("empty.txt", b"", 422, "unprocessable_document"),
    ],
)
def test_bad_uploads_are_rejected_cleanly(
    api: TestClient, filename: str, data: bytes, status: int, code: str
) -> None:
    collection_id = _create_collection(api)

    result = _upload(api, collection_id, filename, data)

    assert result["status"] == status
    assert result["body"]["error"]["code"] == code
    assert _db_counts() == (0, 0)


def test_oversized_upload_is_413(api: TestClient) -> None:
    collection_id = _create_collection(api)

    result = _upload(api, collection_id, "big.txt", b"a" * (1024 * 1024 + 1))  # limit is 1 MB

    assert result["status"] == 413


def test_upload_to_missing_collection_is_404(api: TestClient) -> None:
    result = _upload(api, str(uuid.uuid4()), "policy.md", MARKDOWN)

    assert result["status"] == 404


def test_embedding_failure_is_502_and_leaves_nothing_behind(
    api: TestClient, fake_embedder: FakeEmbedder
) -> None:
    """Atomicity: if embedding fails, no document and no chunks are saved."""
    collection_id = _create_collection(api)
    fake_embedder.fail = True

    result = _upload(api, collection_id, "policy.md", MARKDOWN)

    assert result["status"] == 502
    assert result["body"]["error"]["code"] == "embedding_provider_error"
    assert _db_counts() == (0, 0)

    # And the same file can be uploaded successfully once the provider recovers.
    fake_embedder.fail = False
    assert _upload(api, collection_id, "policy.md", MARKDOWN)["status"] == 201
