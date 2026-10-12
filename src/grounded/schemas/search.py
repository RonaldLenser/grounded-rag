"""API contracts for retrieval (search)."""

import uuid

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=1000,
        examples=["What does the GOVERN function cover?"],
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Number of chunks to return (default 6)"
    )


class SearchResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    page_start: int | None
    page_end: int | None
    heading: str | None
    content: str
    score: float = Field(description="Cosine similarity, higher is more relevant")


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
