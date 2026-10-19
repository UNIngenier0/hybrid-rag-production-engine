"""Request and response models for the HTTP API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from rag_engine.core.health import ReadinessReport
from rag_engine.core.models import RetrievalMode, RetrievalResult


class LivenessResponse(BaseModel):
    status: Literal["ok"]


class ProbeStatus(BaseModel):
    ok: bool
    latency_ms: float
    error: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, ProbeStatus]

    @classmethod
    def from_report(cls, report: ReadinessReport) -> "ReadinessResponse":
        return cls(
            status="ready" if report.ready else "not_ready",
            checks={
                result.name: ProbeStatus(
                    ok=result.ok, latency_ms=result.latency_ms, error=result.error
                )
                for result in report.results
            },
        )


class SearchFiltersRequest(BaseModel):
    """Both lists are optional; an empty one does not restrict the search."""

    source_types: list[str] = Field(default_factory=list, max_length=20)
    doc_ids: list[str] = Field(default_factory=list, max_length=100)


class SearchRequest(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    mode: RetrievalMode = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    filters: SearchFiltersRequest = Field(default_factory=SearchFiltersRequest)
    # Attach each chunk's larger parent passage, the text meant for the LLM.
    include_context: bool = False


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    section: str | None
    text: str
    context: str | None
    score: float
    dense_rank: int | None
    bm25_rank: int | None
    fusion_rank: int | None


class SearchResponse(BaseModel):
    query: str
    mode: RetrievalMode
    hits: list[SearchHit]
    # Distinct documents in the order their best chunk appears.
    document_ids: list[str]
    timings_ms: dict[str, float]

    @classmethod
    def from_result(cls, result: RetrievalResult) -> "SearchResponse":
        return cls(
            query=result.query,
            mode=result.mode,
            hits=[
                SearchHit(
                    rank=rank,
                    chunk_id=hit.point_id,
                    doc_id=hit.doc_id,
                    source_type=hit.source_type,
                    title=hit.title,
                    section=hit.section,
                    text=hit.text,
                    context=hit.context,
                    score=hit.score,
                    dense_rank=hit.dense_rank,
                    bm25_rank=hit.bm25_rank,
                    fusion_rank=hit.fusion_rank,
                )
                for rank, hit in enumerate(result.hits, start=1)
            ],
            document_ids=result.document_ids,
            timings_ms=result.timings_ms,
        )
