"""All SQL lives here. Services call these functions; they never build queries.

Keeping queries in one place means one file to review for performance
(indexes, N+1 queries) and for security (everything is parameterized).
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from grounded.db.models import Chunk, Collection, Document


# ----- Collections -----
async def create_collection(session: AsyncSession, name: str) -> Collection:
    collection = Collection(name=name)
    session.add(collection)
    await session.flush()
    return collection


async def get_collection(session: AsyncSession, collection_id: uuid.UUID) -> Collection | None:
    return await session.get(Collection, collection_id)


async def list_collections(session: AsyncSession) -> Sequence[Collection]:
    result = await session.scalars(select(Collection).order_by(Collection.created_at))
    return result.all()


# ----- Documents -----
async def get_document(session: AsyncSession, document_id: uuid.UUID) -> Document | None:
    return await session.get(Document, document_id)


async def find_document_by_hash(
    session: AsyncSession, collection_id: uuid.UUID, sha256: str
) -> Document | None:
    # Served by the unique index on (collection_id, sha256).
    return await session.scalar(
        select(Document).where(Document.collection_id == collection_id, Document.sha256 == sha256)
    )


async def list_documents_with_chunk_counts(
    session: AsyncSession, collection_id: uuid.UUID
) -> list[tuple[Document, int]]:
    # One query with a LEFT JOIN + GROUP BY, not one COUNT query per document (N+1).
    statement = (
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .where(Document.collection_id == collection_id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    )
    rows = await session.execute(statement)
    return [(document, count) for document, count in rows]


async def count_chunks(session: AsyncSession, document_id: uuid.UUID) -> int:
    count = await session.scalar(
        select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
    )
    return count or 0


async def delete_document(session: AsyncSession, document: Document) -> None:
    # Chunks are removed by the database (ON DELETE CASCADE), not row by row in Python.
    await session.delete(document)
    await session.flush()
