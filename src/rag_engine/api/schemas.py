"""Request and response models for the HTTP API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from rag_engine.core.health import ReadinessReport
from rag_engine.core.models import RetrievalMode, RetrievalResult
from rag_engine.generation.citations import cited_numbers
from rag_engine.generation.query import QueryResult


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


class QueryRequest(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    # Left out, the server picks `hybrid+rerank` when a reranker is configured and `hybrid` if not.
    mode: RetrievalMode | None = None
    # Passages the answer is written from; the server's QUERY_CHUNKS when left out.
    chunks: int | None = Field(default=None, ge=1, le=10)
    filters: SearchFiltersRequest = Field(default_factory=SearchFiltersRequest)
    # Run the judge that checks each citation; the server's VERIFY_CITATIONS when left out.
    verify: bool | None = None


class CitedPassage(BaseModel):
    marker: int
    doc_id: str
    title: str
    source_type: str
    # True when every statement that cites this passage was verified, False when one was not, and
    # null when no judge ran.
    verified: bool | None
    # Positions in `claims` of the statements that cite it.
    claims: list[int]


class ClaimResult(BaseModel):
    text: str
    markers: list[int]
    status: str
    reason: str | None
    # The sentence of the cited passage that the judge pointed at, found in that passage.
    evidence: str | None


class VerificationResult(BaseModel):
    method: Literal["judge", "values"]
    model: str | None
    statements: int
    verified: int
    all_verified: bool
    error: str | None


class Usage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    # null when the model has no known price
    cost_usd: float | None


class QueryResponse(BaseModel):
    query: str
    mode: RetrievalMode
    answer: str
    # The answer exactly as the model wrote it, with its [n] markers.
    answer_with_markers: str
    citations: list[CitedPassage]
    claims: list[ClaimResult]
    verification: VerificationResult
    # Documents the answer cites, in citation order.
    document_ids: list[str]
    usage: Usage
    timings_ms: dict[str, float]

    @classmethod
    def from_result(cls, result: QueryResult) -> "QueryResponse":
        answer, verification = result.answer, result.verification
        blocks = {block.number: block for block in answer.blocks}
        judged = verification.method == "judge"
        citations = []
        for marker in (n for n in cited_numbers(answer.raw_text) if n in blocks):
            citing = [i for i, claim in enumerate(verification.claims) if marker in claim.markers]
            verified = (
                all(verification.claims[i].status == "verified" for i in citing)
                if judged and citing
                else None
            )
            block = blocks[marker]
            citations.append(
                CitedPassage(
                    marker=marker,
                    doc_id=block.doc_id,
                    title=block.title,
                    source_type=block.source_type,
                    verified=verified,
                    claims=citing,
                )
            )
        return cls(
            query=result.question,
            mode=answer.mode,
            answer=answer.text,
            answer_with_markers=answer.raw_text,
            citations=citations,
            claims=[
                ClaimResult(
                    text=claim.text,
                    markers=list(claim.markers),
                    status=claim.status,
                    reason=claim.reason,
                    evidence=claim.quote,
                )
                for claim in verification.claims
            ],
            verification=VerificationResult(
                method=verification.method,
                model=verification.model,
                statements=len(verification.checked),
                verified=verification.verified,
                all_verified=verification.all_verified,
                error=verification.error,
            ),
            document_ids=answer.cited_doc_ids,
            usage=Usage(
                model=answer.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
            ),
            timings_ms=result.timings_ms,
        )
