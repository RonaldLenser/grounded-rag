"""API contracts for retrieval (search)."""

import uuid
from typing import Literal

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
    vector_rank: int | None = Field(
        default=None, description="Hybrid only: rank in the vector results"
    )
    keyword_rank: int | None = Field(
        default=None, description="Hybrid only: rank in the keyword results"
    )
    fused_score: float | None = Field(
        default=None, description="Hybrid only: Reciprocal Rank Fusion score (results order)"
    )


class SearchResponse(BaseModel):
    query: str
    strategy: Literal["vector", "hybrid"]
    results: list[SearchResult]
