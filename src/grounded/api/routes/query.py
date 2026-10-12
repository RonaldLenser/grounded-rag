"""Question answering endpoint: the main feature of Grounded."""

import uuid

from fastapi import APIRouter, Depends

from grounded.api.deps import AnswererDep
from grounded.api.rate_limit import rate_limit
from grounded.retrieval.retriever import best_similarity
from grounded.schemas.collections import ErrorResponse
from grounded.schemas.query import (
    Citation,
    Claim,
    QueryRequest,
    QueryResponse,
    RetrievalInfo,
    Usage,
    VerificationInfo,
)

router = APIRouter(prefix="/v1", tags=["query"])


@router.post(
    "/collections/{collection_id}/query",
    response_model=QueryResponse,
    responses={code: {"model": ErrorResponse} for code in (404, 422, 429, 502, 503)},
    dependencies=[Depends(rate_limit("query"))],
)
async def query(
    collection_id: uuid.UUID, body: QueryRequest, answerer: AnswererDep
) -> QueryResponse:
    result = await answerer.answer(collection_id, body.question, top_k=body.top_k)
    verification = result.verification
    return QueryResponse(
        query_id=result.query_id,
        status=verification.status,
        answer=verification.answer,
        notice=verification.notice,
        claims=[Claim(text=c.text, source_ids=c.source_ids) for c in verification.claims],
        citations=[
            Citation(
                source_id=c.source.source_id,
                quote=c.quote,
                document_id=c.source.chunk.document_id,
                chunk_id=c.source.chunk.chunk_id,
                filename=c.source.chunk.filename,
                page_start=c.source.chunk.page_start,
                page_end=c.source.chunk.page_end,
                heading=c.source.chunk.heading,
            )
            for c in verification.verified
        ],
        retrieval=RetrievalInfo(
            strategy=result.strategy,
            chunks_retrieved=len(result.retrieved),
            top_score=best_similarity(result.retrieved),
        ),
        verification=VerificationInfo(
            claims_supported=len(verification.claims),
            claims_rejected=len(verification.rejected_claims),
            citations_verified=len(verification.verified),
            citations_rejected=len(verification.rejected_evidence),
            downgraded_from=verification.downgraded_from,
        ),
        usage=Usage(
            llm_called=result.llm_called,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        ),
        latency_ms=result.latency_ms,
    )
