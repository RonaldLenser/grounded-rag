"""Retrieval: question in, most relevant chunks out.

Vector (always runs):
1. Embed the question with the SAME model used for the documents. Vectors
   from different models live in different spaces and can't be compared.
2. Ask Postgres for the nearest chunk vectors (cosine similarity, HNSW index).

Hybrid (RETRIEVAL_STRATEGY=hybrid) adds:
3. IDF-weighted keyword search over the same collection (exact terms, rare
   words, section names like "MANAGE 1.4" that embeddings blur).
4. Reciprocal Rank Fusion of the two ranked lists (fusion.py).

Every returned chunk carries its cosine `score`, whichever list found it. The
relevance gate uses the BEST cosine score among the results, so its calibrated
threshold means the same thing in both modes.

This module only finds candidates. Deciding whether they're good enough to
answer from is the answerer's job.
"""

import time
import uuid
from dataclasses import dataclass, replace
from typing import Literal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from grounded.config import Settings
from grounded.db import repositories
from grounded.db.models import Chunk
from grounded.errors import NotFoundError
from grounded.providers.base import Embedder
from grounded.retrieval.fusion import reciprocal_rank_fusion

logger = structlog.get_logger(__name__)

Strategy = Literal["vector", "hybrid"]


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
    score: float  # cosine similarity to the question: higher = more relevant
    # Hybrid only (None in vector mode): where each method ranked this chunk.
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None


def best_similarity(chunks: list[RetrievedChunk]) -> float | None:
    """The relevance gate's input: the best cosine score, regardless of result order."""
    return max((chunk.score for chunk in chunks), default=None)


class Retriever:
    def __init__(self, session: AsyncSession, embedder: Embedder, settings: Settings) -> None:
        self._session = session
        self._embedder = embedder
        self._settings = settings

    @property
    def strategy(self) -> Strategy:
        return self._settings.retrieval_strategy

    async def search(
        self, collection_id: uuid.UUID, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        top_k = top_k or self._settings.retrieval_top_k
        if await repositories.get_collection(self._session, collection_id) is None:
            raise NotFoundError(f"Collection {collection_id} not found")

        started = time.perf_counter()
        [query_vector] = await self._embedder.embed([query])
        embedded = time.perf_counter()

        if self.strategy == "hybrid":
            results = await self._hybrid(collection_id, query, query_vector, top_k)
        else:
            rows = await self._vector(collection_id, query_vector, top_k)
            results = [_to_result(chunk, filename, score) for chunk, filename, score in rows]

        # Log scores, never the query text: user questions can contain private data.
        # (The database does keep the question in query_logs; see "Data retention"
        # in DESIGN.md. Logs go to a wider audience, so they never get it.)
        top = best_similarity(results)
        logger.info(
            "retrieval_completed",
            collection_id=str(collection_id),
            strategy=self.strategy,
            top_k=top_k,
            results=len(results),
            top_score=round(top, 4) if top is not None else None,
            keyword_only=sum(r.vector_rank is None and r.keyword_rank is not None for r in results),
            embed_ms=round((embedded - started) * 1000, 1),
            search_ms=round((time.perf_counter() - embedded) * 1000, 1),
        )
        return results

    async def _vector(
        self, collection_id: uuid.UUID, query_vector: list[float], limit: int
    ) -> list[tuple[Chunk, str, float]]:
        return await repositories.search_chunks_by_vector(
            self._session,
            collection_id,
            query_vector,
            top_k=limit,
            ef_search=self._settings.hnsw_ef_search,
        )

    async def _hybrid(
        self, collection_id: uuid.UUID, query: str, query_vector: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        candidates = max(self._settings.hybrid_candidates, top_k)
        vector_rows = await self._vector(collection_id, query_vector, candidates)
        keyword_rows = await repositories.search_chunk_ids_by_keywords(
            self._session, collection_id, query, candidates
        )

        fused = reciprocal_rank_fusion(
            [[chunk.id for chunk, _, _ in vector_rows], [chunk_id for chunk_id, _ in keyword_rows]],
            k=self._settings.rrf_k,
        )[:top_k]

        # Chunks found only by keyword search still need their row and cosine score.
        loaded = {chunk.id: (chunk, filename, score) for chunk, filename, score in vector_rows}
        missing = [f.item_id for f in fused if f.item_id not in loaded]
        for chunk, filename, score in await repositories.get_chunks_with_similarity(
            self._session, missing, query_vector
        ):
            loaded[chunk.id] = (chunk, filename, score)

        results = []
        for item in fused:
            if item.item_id not in loaded:  # deleted between queries: skip, don't fail
                continue
            chunk, filename, score = loaded[item.item_id]
            vector_rank, keyword_rank = item.ranks
            results.append(
                replace(
                    _to_result(chunk, filename, score),
                    vector_rank=vector_rank,
                    keyword_rank=keyword_rank,
                    fused_score=item.score,
                )
            )
        return results


def _to_result(chunk: Chunk, filename: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
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
