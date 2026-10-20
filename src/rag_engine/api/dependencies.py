"""FastAPI dependencies that expose the objects wired in `create_app`."""

from fastapi import HTTPException, Request, status

from rag_engine.core.config import Settings
from rag_engine.core.ports import HealthProbe, Retriever
from rag_engine.generation.query import QueryService


def get_probes(request: Request) -> tuple[HealthProbe, ...]:
    probes: tuple[HealthProbe, ...] = request.app.state.probes
    return probes


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_retriever(request: Request) -> Retriever:
    retriever: Retriever = request.app.state.retriever
    return retriever


def get_query_service(request: Request) -> QueryService:
    service: QueryService | None = request.app.state.query_service
    if service is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "answer generation is not configured: set ANTHROPIC_API_KEY",
        )
    return service
