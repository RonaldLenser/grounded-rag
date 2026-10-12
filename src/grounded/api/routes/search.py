"""Retrieval-only endpoint: see exactly which chunks a question would use.

Useful on its own for debugging ("why did it answer that?") and for
retrieval evals (M7), which measure search quality separately from the LLM.
"""

import uuid
from dataclasses import asdict

from fastapi import APIRouter

from grounded.api.deps import RetrieverDep
from grounded.schemas.collections import ErrorResponse
from grounded.schemas.search import SearchRequest, SearchResponse, SearchResult

router = APIRouter(prefix="/v1", tags=["search"])


@router.post(
    "/collections/{collection_id}/search",
    response_model=SearchResponse,
    responses={code: {"model": ErrorResponse} for code in (404, 422, 502, 503)},
)
async def search(
    collection_id: uuid.UUID, body: SearchRequest, retriever: RetrieverDep
) -> SearchResponse:
    results = await retriever.search(collection_id, body.query, top_k=body.top_k)
    return SearchResponse(
        query=body.query,
        results=[SearchResult(**asdict(result)) for result in results],
    )
