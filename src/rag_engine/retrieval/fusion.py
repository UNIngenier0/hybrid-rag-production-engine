"""Reciprocal Rank Fusion: merge the dense and the lexical rankings, comparing only their ranks."""

from collections.abc import Sequence
from dataclasses import replace

from rag_engine.core.models import ChunkHit


def reciprocal_rank_fusion(
    dense: Sequence[ChunkHit],
    sparse: Sequence[ChunkHit],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[ChunkHit]:
    """Fuse two ranked lists of chunks into one, best first.

    A chunk scores `weight / (k + rank)` in each list that returned it (rank 1 is best), and the
    scores add up, so a chunk found by both retrievers beats one found by a single retriever at
    the same rank. Cosine similarity and BM25 scores live on different scales, which is why only
    the ranks are used. Chunks are identified by their point, so two chunks of one document stay
    separate. Ties keep the order in which the chunks were first seen, dense list first.
    """
    fused: dict[str, ChunkHit] = {}
    for hits, weight, is_dense in ((dense, dense_weight, True), (sparse, sparse_weight, False)):
        for rank, hit in enumerate(hits, start=1):
            seen = fused.get(hit.point_id, replace(hit, score=0.0))
            fused[hit.point_id] = replace(
                seen,
                score=seen.score + weight / (k + rank),
                dense_rank=rank if is_dense else seen.dense_rank,
                bm25_rank=seen.bm25_rank if is_dense else rank,
            )
    return sorted(fused.values(), key=lambda hit: hit.score, reverse=True)
