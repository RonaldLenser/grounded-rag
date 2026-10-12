"""API contracts for collections and documents.

These Pydantic models define exactly what clients send and receive. They are
separate from the database models on purpose: the API never leaks internal
columns (like sha256 or embeddings) and the DB schema can change without
breaking clients.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CollectionCreate(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9 _.-]*$",
        examples=["nist-ai-rmf"],
    )


class CollectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    collection_id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    page_count: int | None
    status: str
    chunk_count: int
    created_at: datetime


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
