"""Integration tests against a real PostgreSQL + pgvector database.

These prove the things mocks can't: the migration actually runs (see the
`migrated` fixture in conftest.py), the schema matches the models, and
Postgres enforces our constraints.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from grounded.db.models import EMBEDDING_DIM, Base, Chunk, Collection, Document


def run_with_engine[T](fn: Callable[[AsyncEngine], Awaitable[T]]) -> T:
    """Run an async DB check from a plain (sync) test with a throwaway engine."""

    async def runner() -> T:
        engine = create_async_engine(os.environ["DATABASE_URL"])
        try:
            return await fn(engine)
        finally:
            await engine.dispose()

    return asyncio.run(runner())


def test_ready_returns_ok_with_real_database(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "ok"}


def test_models_match_migrations() -> None:
    """Fails if someone changes models.py without writing a migration (or vice versa)."""

    async def diff(engine: AsyncEngine) -> list[Any]:
        async with engine.connect() as conn:
            return await conn.run_sync(
                lambda sync_conn: compare_metadata(
                    MigrationContext.configure(sync_conn), Base.metadata
                )
            )

    assert run_with_engine(diff) == []


def test_pgvector_extension_and_indexes_exist() -> None:
    async def inspect(engine: AsyncEngine) -> tuple[str | None, set[str]]:
        async with engine.connect() as conn:
            ext = await conn.scalar(text("SELECT extname FROM pg_extension WHERE extname='vector'"))
            rows = await conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename='chunks'")
            )
            return ext, {row[0] for row in rows}

    extension, indexes = run_with_engine(inspect)

    assert extension == "vector"
    assert {"ix_chunks_embedding_hnsw", "ix_chunks_tsv"} <= indexes


async def _seed_document(session: AsyncSession, sha: str = "a" * 64) -> Document:
    collection = Collection(name=f"test-{sha[:8]}")
    document = Document(
        collection=collection,
        filename="policy.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        sha256=sha,
    )
    session.add(document)
    await session.flush()
    return document


def test_chunk_round_trip_with_vector_and_generated_tsv() -> None:
    async def scenario(engine: AsyncEngine) -> tuple[list[float], str]:
        async with AsyncSession(engine) as session:
            document = await _seed_document(session, sha="b" * 64)
            embedding = [0.0] * EMBEDDING_DIM
            embedding[0] = 1.0
            session.add(
                Chunk(
                    document=document,
                    chunk_index=0,
                    content="Risk management frameworks govern AI systems.",
                    token_count=7,
                    page_start=1,
                    page_end=1,
                    embedding=embedding,
                )
            )
            await session.flush()
            row = (await session.execute(select(Chunk.embedding, Chunk.tsv))).one()
            await session.rollback()
            return list(row.embedding), str(row.tsv)

    stored_embedding, tsv = run_with_engine(scenario)

    assert len(stored_embedding) == EMBEDDING_DIM
    assert stored_embedding[0] == 1.0
    # Postgres generated the keyword index itself, with stemming ("frameworks" -> "framework").
    assert "framework" in tsv


def test_duplicate_upload_in_same_collection_is_rejected() -> None:
    async def scenario(engine: AsyncEngine) -> None:
        async with AsyncSession(engine) as session:
            document = await _seed_document(session, sha="c" * 64)
            session.add(
                Document(
                    collection_id=document.collection_id,
                    filename="same-file-renamed.pdf",
                    content_type="application/pdf",
                    size_bytes=1024,
                    sha256="c" * 64,
                )
            )
            with pytest.raises(IntegrityError, match="uq_documents_collection_id_sha256"):
                await session.flush()
            await session.rollback()

    run_with_engine(scenario)


def test_invalid_document_status_is_rejected() -> None:
    async def scenario(engine: AsyncEngine) -> None:
        async with AsyncSession(engine) as session:
            document = await _seed_document(session, sha="d" * 64)
            document.status = "banana"
            with pytest.raises(IntegrityError, match="ck_documents_status"):
                await session.flush()
            await session.rollback()

    run_with_engine(scenario)


def test_deleting_collection_cascades_to_documents_and_chunks() -> None:
    async def scenario(engine: AsyncEngine) -> tuple[bool, bool]:
        async with AsyncSession(engine) as session:
            document = await _seed_document(session, sha="e" * 64)
            session.add(
                Chunk(
                    document=document,
                    chunk_index=0,
                    content="text",
                    token_count=1,
                    embedding=[0.1] * EMBEDDING_DIM,
                )
            )
            await session.flush()
            # Delete with SQL (not the ORM) to prove Postgres itself enforces the cascade.
            await session.execute(delete(Collection).where(Collection.id == document.collection_id))
            docs = await session.scalar(
                select(Document.id).where(Document.id == document.id).limit(1)
            )
            chunks = await session.scalar(
                select(Chunk.id).where(Chunk.document_id == document.id).limit(1)
            )
            await session.rollback()
            return docs is not None, chunks is not None

    docs_left, chunks_left = run_with_engine(scenario)

    assert not docs_left
    assert not chunks_left
