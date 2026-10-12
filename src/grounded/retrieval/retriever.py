"""Retrieval: question in, most relevant chunks out.

Steps:
1. Embed the question with the SAME model used for the documents. Vectors
   from different models live in different spaces and can't be compared.
2. Ask Postgres for the nearest chunk vectors (cosine similarity, HNSW index).
3. Return chunks with their source info (file, pages) and similarity score.

This module only finds candidates. Deciding whether they're good enough to
answer from (the relevance threshold) is the answer step's job in M4.
"""

import time
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from grounded.config import Settings
from grounded.db import repositories
from grounded.errors import NotFoundError
from grounded.providers.base import Embedder

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    content: str
    page_start: int | None
    page_end: int | None
    heading: str | None
    score: float  # cosine similarity: higher = more relevant


class Retriever:
    def __init__(self, session: AsyncSession, embedder: Embedder, settings: Settings) -> None:
        self._session = session
        self._embedder = embedder
        self._settings = settings

    async def search(
        self, collection_id: uuid.UUID, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        top_k = top_k or self._settings.retrieval_top_k
        if await repositories.get_collection(self._session, collection_id) is None:
            raise NotFoundError(f"Collection {collection_id} not found")

        started = time.perf_counter()
        [query_vector] = await self._embedder.embed([query])
        embedded = time.perf_counter()

        rows = await repositories.search_chunks_by_vector(
            self._session,
            collection_id,
            query_vector,
            top_k=top_k,
            ef_search=self._settings.hnsw_ef_search,
        )
        results = [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                heading=chunk.heading,
                score=score,
            )
            for chunk, filename, score in rows
        ]

        # Log scores, never the query text: user questions can contain private data.
        logger.info(
            "retrieval_completed",
            collection_id=str(collection_id),
            top_k=top_k,
            results=len(results),
            top_score=round(results[0].score, 4) if results else None,
            embed_ms=round((embedded - started) * 1000, 1),
            search_ms=round((time.perf_counter() - embedded) * 1000, 1),
        )
        return results
