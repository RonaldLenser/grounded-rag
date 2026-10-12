"""All SQL lives here. Services call these functions; they never build queries.

Keeping queries in one place means one file to review for performance
(indexes, N+1 queries) and for security (everything is parameterized).
"""

import uuid
from collections.abc import Sequence
from datetime import date

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from grounded.db.models import Chunk, Collection, Document, LLMDailyUsage, QueryLog


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


async def delete_collection(session: AsyncSession, collection: Collection) -> None:
    # Documents, chunks and query logs go with it (ON DELETE CASCADE in the database).
    await session.delete(collection)
    await session.flush()


async def delete_document(session: AsyncSession, document: Document) -> None:
    # Chunks are removed by the database (ON DELETE CASCADE), not row by row in Python.
    await session.delete(document)
    await session.flush()


# ----- Vector search -----
async def search_chunks_by_vector(
    session: AsyncSession,
    collection_id: uuid.UUID,
    query_vector: list[float],
    top_k: int,
    ef_search: int,
) -> list[tuple[Chunk, str, float]]:
    """Nearest chunks to `query_vector` within one collection, best first.

    Returns (chunk, filename, cosine_similarity). Similarity = 1 - cosine distance:
    1.0 = same direction (very similar meaning), ~0 = unrelated.
    """
    # These settings only last for this transaction, so they can't leak to other requests.
    # ef_search: how many candidates the HNSW graph walk keeps (recall vs speed).
    # set_config(..., is_local => true) is the parameterized form of SET LOCAL.
    await session.execute(
        text("SELECT set_config('hnsw.ef_search', :value, true)"), {"value": str(ef_search)}
    )
    # Iterative scan (pgvector >= 0.8): HNSW finds nearest neighbors across ALL
    # collections first, then our WHERE filter removes other collections. Without
    # this, a small collection could get fewer than top_k results. With it, the
    # index keeps scanning until enough rows survive the filter.
    await session.execute(text("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)"))

    distance = Chunk.embedding.cosine_distance(query_vector)
    statement = (
        select(Chunk, Document.filename, (1 - distance).label("similarity"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.collection_id == collection_id)
        # Only answer from fully ingested documents. Ingestion is atomic today, so
        # non-ready chunks shouldn't exist; this keeps it true if that ever changes.
        .where(Document.status == "ready")
        # ORDER BY distance + LIMIT is the exact shape that lets Postgres use the HNSW index.
        .order_by(distance)
        .limit(top_k)
    )
    rows = await session.execute(statement)
    return [(chunk, filename, float(similarity)) for chunk, filename, similarity in rows]


# ----- Keyword search -----
# IDF-weighted term matching: a chunk's score is the sum, over the question's
# words it contains, of ln(N / df) (N = chunks in the collection, df = chunks
# containing the word). Rare words ("residual", "Dioptra") count a lot; words on
# every page ("AI", "risk") count ~0. This is BM25's core idea without its
# term-frequency and length terms (chunks are all ~500 tokens, so length
# normalization buys little here).
#
# Why not Postgres's built-in ranking: websearch/plainto_tsquery require EVERY
# word to match (natural-language questions rarely do), and ts_rank ignores word
# rarity. On the golden set those found the right page in the top 6 for 11/24
# and 19/24 questions; this query finds 22/24 (docs/DESIGN.md, M8).
#
# Words are stemmed with the same 'english' config as the stored tsv column.
# The OR-tsquery prefilter lets Postgres use the GIN index on chunks.tsv.
_KEYWORD_SEARCH_SQL = text(
    """
    WITH scope AS (
        SELECT c.id, c.tsv
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.collection_id = :collection_id AND d.status = 'ready'
    ),
    terms AS (
        SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('english', :query))) AS lexeme
    ),
    hits AS (
        SELECT s.id AS chunk_id, t.lexeme
        FROM scope s
        CROSS JOIN LATERAL unnest(tsvector_to_array(s.tsv)) AS t(lexeme)
        WHERE s.tsv @@ replace(plainto_tsquery('english', :query)::text, ' & ', ' | ')::tsquery
          AND t.lexeme IN (SELECT lexeme FROM terms)
    ),
    weights AS (
        SELECT lexeme, ln((SELECT count(*) FROM scope)::float8 / count(*)) AS idf
        FROM hits
        GROUP BY lexeme
    )
    SELECT h.chunk_id, sum(w.idf) AS score
    FROM hits h JOIN weights w USING (lexeme)
    GROUP BY h.chunk_id
    HAVING sum(w.idf) > 0
    ORDER BY score DESC, h.chunk_id
    LIMIT :limit
    """
)


async def search_chunk_ids_by_keywords(
    session: AsyncSession, collection_id: uuid.UUID, query: str, limit: int
) -> list[tuple[uuid.UUID, float]]:
    """(chunk_id, keyword score), best first. Empty if no informative word matches."""
    rows = await session.execute(
        _KEYWORD_SEARCH_SQL,
        {"collection_id": collection_id, "query": query, "limit": limit},
    )
    return [(chunk_id, float(score)) for chunk_id, score in rows]


async def get_chunks_with_similarity(
    session: AsyncSession, chunk_ids: list[uuid.UUID], query_vector: list[float]
) -> list[tuple[Chunk, str, float]]:
    """Load chunks by id with their cosine similarity to the query (order not guaranteed)."""
    if not chunk_ids:
        return []
    distance = Chunk.embedding.cosine_distance(query_vector)
    statement = (
        select(Chunk, Document.filename, (1 - distance).label("similarity"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(chunk_ids))
    )
    rows = await session.execute(statement)
    return [(chunk, filename, float(similarity)) for chunk, filename, similarity in rows]


# ----- Query logs and LLM budget -----
async def add_query_log(
    session: AsyncSession,
    *,
    collection_id: uuid.UUID,
    question: str,
    status: str,
    retrieved_chunk_ids: list[uuid.UUID],
    scores: list[float],
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: int,
) -> QueryLog:
    log = QueryLog(
        collection_id=collection_id,
        question=question,
        status=status,
        retrieved_chunk_ids=retrieved_chunk_ids,
        scores=scores,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )
    session.add(log)
    await session.flush()
    return log


async def reserve_llm_call(session: AsyncSession, day: date, limit: int) -> bool:
    """Atomically claim one LLM call from `day`'s budget. False = budget used up.

    One statement, no read-then-write race:
      INSERT (day, 1) ... ON CONFLICT (day)
      DO UPDATE SET calls = calls + 1 WHERE calls < limit
      RETURNING calls
    Postgres row-locks the day's counter while it checks and increments, so two
    concurrent requests can't both take the last slot. When the WHERE fails, no
    row is returned and nothing changes.

    The caller must COMMIT right away (before the slow LLM call) so the row lock
    is held for milliseconds, not seconds.
    """
    statement = (
        pg_insert(LLMDailyUsage)
        .values(day=day, calls=1)
        .on_conflict_do_update(
            index_elements=[LLMDailyUsage.day],
            set_={"calls": LLMDailyUsage.calls + 1, "updated_at": func.now()},
            where=LLMDailyUsage.calls < limit,
        )
        .returning(LLMDailyUsage.calls)
    )
    return await session.scalar(statement) is not None
