"""Answers: a cited answer to a question, with each citation checked against its passage."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from rag_engine.api.dependencies import get_app_settings, get_query_service
from rag_engine.api.schemas import QueryRequest, QueryResponse
from rag_engine.core.config import Settings
from rag_engine.core.errors import BudgetExceededError, LLMError, RetrievalUnavailableError
from rag_engine.core.models import SearchFilters
from rag_engine.generation.query import QueryService

router = APIRouter(prefix="/v1", tags=["query"])


@router.post(
    "/query",
    responses={
        502: {"description": "The language model could not answer"},
        503: {"description": "Retrieval or answer generation is not available"},
    },
)
async def query(
    request: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> QueryResponse:
    """Retrieve the best passages, write an answer that cites them, and verify each citation.

    The answer comes back twice, clean and with its `[n]` markers, next to the statements it was
    split into and what was found for each: verified against the cited passage, only partly
    supported, unsupported, or not cited at all.
    """
    filters = SearchFilters(
        source_types=tuple(request.filters.source_types), doc_ids=tuple(request.filters.doc_ids)
    )
    verify = settings.verify_citations if request.verify is None else request.verify
    try:
        result = await service.query(
            request.query,
            mode=request.mode,
            chunks=request.chunks,
            filters=filters,
            verify=verify,
        )
    except RetrievalUnavailableError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "retrieval unavailable") from error
    except BudgetExceededError as error:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "the model budget is spent"
        ) from error
    except LLMError as error:
        # The cause is in the server logs; the client only learns that the model failed.
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "the language model failed") from error
    return QueryResponse.from_result(result)
