"""Database tables, defined as typed SQLAlchemy models.

Four tables:
  collections  a named group of documents (e.g. the demo corpus vs. eval corpus)
  documents    one uploaded file and its processing status
  chunks       a piece of a document's text plus its embedding vector
  query_logs   one row per question asked (for debugging, cost, and evals)

The schema itself is created by Alembic migrations (migrations/versions/),
not by these classes. A test checks that the two never drift apart.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Must match the embedding model's output size (text-embedding-3-small = 1536).
# Changing embedding models means a new migration and re-embedding every chunk.
EMBEDDING_DIM = 1536

DOCUMENT_STATUSES = ("processing", "ready", "failed")
ANSWER_STATUSES = ("answered", "partial", "insufficient_context")

# Deterministic constraint/index names, so migrations can reference them reliably.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,  # Python knows the ID before insert
        server_default=text("gen_random_uuid()"),  # raw SQL inserts get one too
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    created_at: Mapped[datetime] = _created_at()

    documents: Mapped[list["Document"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan", passive_deletes=True
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # Same file uploaded twice to one collection = same hash = rejected.
        UniqueConstraint("collection_id", "sha256"),
        CheckConstraint(_in_list("status", DOCUMENT_STATUSES), name="status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="processing")
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    collection: Mapped[Collection] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        # HNSW = approximate nearest-neighbor index. Makes vector search fast at
        # the cost of being approximate. vector_cosine_ops = cosine distance.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN index for Postgres full-text (keyword) search, used by hybrid retrieval.
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    heading: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    # Generated by Postgres from `content`; the app never writes it.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )
    created_at: Mapped[datetime] = _created_at()

    document: Mapped[Document] = relationship(back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"
    __table_args__ = (CheckConstraint(_in_list("status", ANSWER_STATUSES), name="status"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    retrieved_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default=text("'{}'")
    )
    scores: Mapped[list[float]] = mapped_column(
        ARRAY(Float), nullable=False, server_default=text("'{}'")
    )
    # Null when we refused before calling the LLM (low retrieval confidence).
    model: Mapped[str | None] = mapped_column(String(100))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = _created_at()
