"""Document ingestion: validate → dedupe → parse → chunk → embed → store.

Key design decision: ingestion is ATOMIC. All slow and failure-prone work
(parsing, chunking, the embedding API call) happens in memory first. Only
when everything has succeeded do we write the document and all its chunks
in a single transaction. A failure at any step leaves nothing behind: no
half-ingested documents, no orphaned chunks, no cleanup jobs.

Tradeoff: the upload request waits for the whole pipeline (seconds for a
large PDF). Fine at this scale. At larger scale this moves to a background
queue, and the documents.status column ('processing' -> 'ready'/'failed')
already exists for that.
"""

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from grounded.config import Settings
from grounded.db import repositories
from grounded.db.models import Chunk, Document
from grounded.errors import (
    DuplicateDocumentError,
    NotFoundError,
    PayloadTooLargeError,
    UnprocessableDocumentError,
)
from grounded.ingestion.chunker import chunk_pages
from grounded.ingestion.parsers import detect_content_type, parse_document
from grounded.providers.base import Embedder

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    chunk_count: int


class IngestionService:
    def __init__(self, session: AsyncSession, embedder: Embedder, settings: Settings) -> None:
        self._session = session
        self._embedder = embedder
        self._settings = settings

    async def ingest(self, collection_id: uuid.UUID, filename: str, data: bytes) -> IngestionResult:
        started = time.perf_counter()
        log = logger.bind(collection_id=str(collection_id), filename=filename, size=len(data))

        # 1. Cheap checks first: fail fast before any expensive work.
        if await repositories.get_collection(self._session, collection_id) is None:
            raise NotFoundError(f"Collection {collection_id} not found")
        if len(data) > self._settings.max_upload_bytes:
            raise PayloadTooLargeError(f"File exceeds {self._settings.max_upload_mb} MB limit")
        if not data:
            raise UnprocessableDocumentError("File is empty")
        content_type = detect_content_type(filename)  # raises 415 for unsupported types

        # 2. Deduplicate by content hash (renaming a file doesn't fool it).
        sha256 = hashlib.sha256(data).hexdigest()
        existing = await repositories.find_document_by_hash(self._session, collection_id, sha256)
        if existing is not None:
            raise DuplicateDocumentError(str(existing.id))

        # 3. Parse + chunk in a worker thread: they're CPU-bound and would
        #    otherwise freeze the event loop for every other request.
        parsed = await asyncio.to_thread(
            parse_document, filename, data, max_pdf_pages=self._settings.max_pdf_pages
        )
        chunks = await asyncio.to_thread(
            chunk_pages,
            parsed.pages,
            chunk_size=self._settings.chunk_size_tokens,
            chunk_overlap=self._settings.chunk_overlap_tokens,
            is_markdown=parsed.is_markdown,
        )
        if not chunks:
            raise UnprocessableDocumentError("Document produced no text chunks")

        # 4. Embed (network call, batched, SDK retries). Nothing is in the DB yet.
        vectors = await self._embedder.embed([chunk.text for chunk in chunks])

        # 5. One transaction: document + every chunk, or nothing.
        document = Document(
            collection_id=collection_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            sha256=sha256,
            page_count=parsed.page_count,
            status="ready",
        )
        document.chunks = [
            Chunk(
                chunk_index=chunk.index,
                content=chunk.text,
                token_count=chunk.token_count,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                heading=chunk.heading,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._session.add(document)
        try:
            await self._session.commit()
        except IntegrityError as exc:
            # Two identical uploads raced past step 2: the unique constraint decides.
            await self._session.rollback()
            winner = await repositories.find_document_by_hash(self._session, collection_id, sha256)
            if winner is None:
                raise
            raise DuplicateDocumentError(str(winner.id)) from exc

        log.info(
            "document_ingested",
            document_id=str(document.id),
            pages=parsed.page_count,
            chunks=len(chunks),
            tokens=sum(chunk.token_count for chunk in chunks),
            embedding_model=self._embedder.model,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return IngestionResult(document=document, chunk_count=len(chunks))
