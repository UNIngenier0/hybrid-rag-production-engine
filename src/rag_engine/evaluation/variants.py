"""Rebuild, from stored candidates, the rankings that the retriever produces.

These use the same fusion function as the service, so a ranking computed here is the ranking the
API would return for the same candidates (a test checks that).
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from rag_engine.core.models import ChunkHit
from rag_engine.evaluation.candidates import QuestionCandidates
from rag_engine.retrieval.fusion import reciprocal_rank_fusion


@dataclass(frozen=True, slots=True)
class FusionSetting:
    """How the two candidate lists are fused: how deep into each, and with which weights."""

    depth: int = 50
    k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0


def dense_ranking(candidates: QuestionCandidates, depth: int = 100) -> list[ChunkHit]:
    return [replace(hit, dense_rank=rank) for rank, hit in enumerate(candidates.dense[:depth], 1)]


def bm25_ranking(candidates: QuestionCandidates, depth: int = 100) -> list[ChunkHit]:
    return [replace(hit, bm25_rank=rank) for rank, hit in enumerate(candidates.bm25[:depth], 1)]


def fused_ranking(
    candidates: QuestionCandidates, setting: FusionSetting | None = None
) -> list[ChunkHit]:
    setting = setting or FusionSetting()
    return reciprocal_rank_fusion(
        candidates.dense[: setting.depth],
        candidates.bm25[: setting.depth],
        k=setting.k,
        dense_weight=setting.dense_weight,
        sparse_weight=setting.sparse_weight,
    )


def reranked_ranking(
    fused: list[ChunkHit], scores: Mapping[str, float], pool: int
) -> list[ChunkHit]:
    """Reorder the first `pool` fused chunks by cross-encoder score (ties keep the fused order)."""
    candidates = fused[:pool]
    missing = [hit.point_id for hit in candidates if hit.point_id not in scores]
    if missing:
        raise KeyError(f"no reranker score for {len(missing)} candidates, such as {missing[0]}")
    ordered = sorted(candidates, key=lambda hit: scores[hit.point_id], reverse=True)
    return [replace(hit, score=scores[hit.point_id]) for hit in ordered]
