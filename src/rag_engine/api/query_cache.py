"""The cache key of a /v1/query: everything that can change the answer, and nothing else."""

import hashlib
import json

from rag_engine.api.schemas import QueryRequest
from rag_engine.core.config import Settings


def cache_key(settings: Settings, request: QueryRequest, verify: bool) -> str:
    """Two requests share a key only when the pipeline would treat them the same: the same
    question (ignoring whitespace), mode, passages, filters and verification, under the same
    models, reranker and abstention threshold. The namespace lets an operator drop every cached
    answer, which is what re-indexing needs."""
    description = {
        "namespace": settings.query_cache_namespace,
        "question": " ".join(request.query.split()),
        "mode": request.mode,
        "chunks": request.chunks or settings.query_chunks,
        "source_types": sorted(request.filters.source_types),
        "doc_ids": sorted(request.filters.doc_ids),
        "verify": verify,
        "llm_model": settings.llm_model,
        "verification_model": settings.verification_model or settings.llm_model,
        "reranker": settings.reranker_model,
        "abstain_threshold": settings.abstain_threshold,
    }
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
