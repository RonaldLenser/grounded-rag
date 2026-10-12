"""Collection and document endpoints."""

import uuid
from typing import Any

from fastapi import APIRouter, File, Response, UploadFile, status
from sqlalchemy.exc import IntegrityError

from grounded.api.deps import AdminDep, IngestionServiceDep, SessionDep, SettingsDep
from grounded.db import repositories
from grounded.errors import ConflictError, NotFoundError, PayloadTooLargeError
from grounded.schemas.collections import (
    CollectionCreate,
    CollectionOut,
    DocumentOut,
    ErrorResponse,
)

router = APIRouter(prefix="/v1", tags=["collections"])

ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 403, 404, 409, 413, 415, 422, 502, 503)
}


@router.post(
    "/collections",
    response_model=CollectionOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[AdminDep],
    responses=ERRORS,
)
async def create_collection(body: CollectionCreate, session: SessionDep) -> CollectionOut:
    try:
        collection = await repositories.create_collection(session, body.name)
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise ConflictError(f"A collection named '{body.name}' already exists") from exc
    return CollectionOut.model_validate(collection)


@router.get("/collections", response_model=list[CollectionOut])
async def list_collections(session: SessionDep) -> list[CollectionOut]:
    return [CollectionOut.model_validate(c) for c in await repositories.list_collections(session)]


@router.delete(
    "/collections/{collection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[AdminDep],
    responses=ERRORS,
)
async def delete_collection(collection_id: uuid.UUID, session: SessionDep) -> Response:
    """Delete a collection with all its documents, chunks and query logs."""
    collection = await repositories.get_collection(session, collection_id)
    if collection is None:
        raise NotFoundError(f"Collection {collection_id} not found")
    await repositories.delete_collection(session, collection)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/collections/{collection_id}/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[AdminDep],
    responses=ERRORS,
)
async def upload_document(
    collection_id: uuid.UUID,
    service: IngestionServiceDep,
    settings: SettingsDep,
    file: UploadFile = File(description="PDF, Markdown (.md), or plain text (.txt)"),  # noqa: B008
) -> DocumentOut:
    # Read at most limit+1 bytes: enough to detect "too big" without loading
    # a huge upload fully into memory.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(f"File exceeds {settings.max_upload_mb} MB limit")

    result = await service.ingest(collection_id, file.filename or "upload", data)
    return DocumentOut.model_validate(
        {**_document_fields(result.document), "chunk_count": result.chunk_count}
    )


@router.get(
    "/collections/{collection_id}/documents",
    response_model=list[DocumentOut],
    responses={404: {"model": ErrorResponse}},
)
async def list_documents(collection_id: uuid.UUID, session: SessionDep) -> list[DocumentOut]:
    if await repositories.get_collection(session, collection_id) is None:
        raise NotFoundError(f"Collection {collection_id} not found")
    rows = await repositories.list_documents_with_chunk_counts(session, collection_id)
    return [
        DocumentOut.model_validate({**_document_fields(doc), "chunk_count": count})
        for doc, count in rows
    ]


@router.get(
    "/documents/{document_id}",
    response_model=DocumentOut,
    responses={404: {"model": ErrorResponse}},
)
async def get_document(document_id: uuid.UUID, session: SessionDep) -> DocumentOut:
    document = await repositories.get_document(session, document_id)
    if document is None:
        raise NotFoundError(f"Document {document_id} not found")
    chunk_count = await repositories.count_chunks(session, document_id)
    return DocumentOut.model_validate({**_document_fields(document), "chunk_count": chunk_count})


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[AdminDep],
    responses=ERRORS,
)
async def delete_document(document_id: uuid.UUID, session: SessionDep) -> Response:
    document = await repositories.get_document(session, document_id)
    if document is None:
        raise NotFoundError(f"Document {document_id} not found")
    await repositories.delete_document(session, document)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _document_fields(document: object) -> dict[str, object]:
    fields = DocumentOut.model_fields.keys() - {"chunk_count"}
    return {name: getattr(document, name) for name in fields}
