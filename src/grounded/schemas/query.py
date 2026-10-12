"""API contracts for question answering."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        examples=["What does the GOVERN function cover?"],
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class Claim(BaseModel):
    text: str = Field(description="One verified statement, without markers")
    source_ids: list[str] = Field(description="Sources whose quotes support this claim")


class Citation(BaseModel):
    source_id: str = Field(description="Marker used in the answer text, e.g. S1")
    quote: str = Field(description="Verified exact text from the source")
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    filename: str
    page_start: int | None
    page_end: int | None
    heading: str | None


class RetrievalInfo(BaseModel):
    chunks_retrieved: int
    top_score: float | None


class VerificationInfo(BaseModel):
    claims_supported: int = Field(description="Claims shown: every quote verified")
    claims_rejected: int = Field(description="Claims dropped: missing or unverifiable evidence")
    citations_verified: int
    citations_rejected: int
    downgraded_from: str | None = Field(
        description="Original model status if verification downgraded it"
    )


class Usage(BaseModel):
    llm_called: bool
    model: str | None
    input_tokens: int | None
    output_tokens: int | None


class QueryResponse(BaseModel):
    query_id: uuid.UUID
    status: Literal["answered", "partial", "insufficient_context"]
    answer: str = Field(description="Built from verified claims only, with [S#] markers")
    notice: str | None = Field(
        description="Shown with partial answers, e.g. when unverifiable statements were left out"
    )
    claims: list[Claim]
    citations: list[Citation]
    retrieval: RetrievalInfo
    verification: VerificationInfo
    usage: Usage
    latency_ms: int
