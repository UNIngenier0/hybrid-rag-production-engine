"""Retrieval-only search: the ranked chunks for a query, without generating an answer."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from rag_engine.api.dependencies import get_retriever
from rag_engine.api.schemas import SearchRequest, SearchResponse
from rag_engine.core.errors import RetrievalUnavailableError
from rag_engine.core.models import SearchFilters
from rag_engine.core.ports import Retriever

router = APIRouter(prefix="/v1", tags=["search"])


@router.post("/search", responses={503: {"description": "Retrieval is not available"}})
async def search(
    request: SearchRequest, retriever: Annotated[Retriever, Depends(get_retriever)]
) -> SearchResponse:
    """Rank the indexed chunks for a query with dense, BM25 or hybrid retrieval."""
    filters = SearchFilters(
        source_types=tuple(request.filters.source_types), doc_ids=tuple(request.filters.doc_ids)
    )
    try:
        result = await retriever.retrieve(
            request.query,
            mode=request.mode,
            filters=filters,
            limit=request.limit,
            include_context=request.include_context,
        )
    except RetrievalUnavailableError as error:
        # The cause is in the server logs; the client only learns that retrieval is down.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "retrieval unavailable") from error
    return SearchResponse.from_result(result)
