"""FastAPI dependencies that expose the objects wired in `create_app`."""

from fastapi import Request

from rag_engine.core.config import Settings
from rag_engine.core.ports import HealthProbe, Retriever


def get_probes(request: Request) -> tuple[HealthProbe, ...]:
    probes: tuple[HealthProbe, ...] = request.app.state.probes
    return probes


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_retriever(request: Request) -> Retriever:
    retriever: Retriever = request.app.state.retriever
    return retriever
