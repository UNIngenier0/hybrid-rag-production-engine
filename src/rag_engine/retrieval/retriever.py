"""The query-side retriever: dense search, BM25 search, fusion, reranking and parent passages."""

import asyncio
import time
from dataclasses import dataclass, replace

from rag_engine.core.errors import RetrievalUnavailableError
from rag_engine.core.models import ChunkHit, RetrievalMode, RetrievalResult, SearchFilters
from rag_engine.core.ports import ChunkSearcher, Embedder, PassageStore, Reranker, SparseEncoder
from rag_engine.indexing.contextual import index_text
from rag_engine.retrieval.fusion import reciprocal_rank_fusion


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    dense_top_k: int = 50
    sparse_top_k: int = 50
    # How many chunks a query returns when the caller does not say.
    default_limit: int = 30
    # Fused chunks the cross-encoder rescored in `hybrid+rerank` mode (at least the limit).
    rerank_top_k: int = 30
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    # BM25 weight when the fused list feeds the reranker (see Settings.rerank_sparse_weight).
    rerank_sparse_weight: float = 1.0


class HybridRetriever:
    """Runs the dense and the BM25 search side by side, fuses them with RRF and can rerank.

    Each branch embeds or encodes the query and searches as soon as its own vector is ready, so
    the slower model does not hold up the other search. Both searches use the same filters.
    Embedding is blocking, so it runs in a worker thread.
    """

    def __init__(
        self,
        searcher: ChunkSearcher,
        embedder: Embedder,
        sparse_encoder: SparseEncoder,
        *,
        passages: PassageStore | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._searcher = searcher
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder
        self._passages = passages
        self._reranker = reranker
        self._config = config or RetrievalConfig()

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        filters: SearchFilters | None = None,
        limit: int | None = None,
        include_context: bool = False,
    ) -> RetrievalResult:
        """Return the best `limit` chunks for `query`, with the time each stage took.

        Every mode returns the same number of chunks, so modes can be compared on equal terms. A
        retriever searches at least `limit` chunks deep, and its configured depth when that is
        larger. In `hybrid+rerank` mode the cross-encoder rescored the first `rerank_top_k` fused
        chunks (or `limit`, if larger) and the best `limit` of them are returned. With
        `include_context`, each hit also carries its parent passage.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        limit = self._config.default_limit if limit is None else limit
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if include_context and self._passages is None:
            raise ValueError("this retriever was built without a passage store")
        if mode == "hybrid+rerank" and self._reranker is None:
            raise RetrievalUnavailableError("no reranker is configured")
        fused_mode = mode in ("hybrid", "hybrid+rerank")
        depth_floor = max(limit, self._config.rerank_top_k) if mode == "hybrid+rerank" else limit

        started = time.perf_counter()
        timings: dict[str, float] = {}
        dense: list[ChunkHit] = []
        sparse: list[ChunkHit] = []
        branches = []
        if mode == "dense" or fused_mode:
            branches.append(self._dense(query, max(self._config.dense_top_k, depth_floor), filters))
        if mode == "bm25" or fused_mode:
            branches.append(
                self._sparse(query, max(self._config.sparse_top_k, depth_floor), filters)
            )
        for name, hits, stage_timings in await asyncio.gather(*branches):
            timings.update(stage_timings)
            if name == "dense":
                dense = hits
            else:
                sparse = hits

        if fused_mode:
            fuse_started = time.perf_counter()
            hits = reciprocal_rank_fusion(
                dense,
                sparse,
                k=self._config.rrf_k,
                dense_weight=self._config.dense_weight,
                sparse_weight=(
                    self._config.rerank_sparse_weight
                    if mode == "hybrid+rerank"
                    else self._config.sparse_weight
                ),
            )
            timings["fuse"] = _since(fuse_started)
            if mode == "hybrid+rerank":
                rerank_started = time.perf_counter()
                hits = await self._rerank(query, hits, limit)
                timings["rerank"] = _since(rerank_started)
        elif mode == "dense":
            hits = [replace(hit, dense_rank=rank) for rank, hit in enumerate(dense, start=1)]
        else:
            hits = [replace(hit, bm25_rank=rank) for rank, hit in enumerate(sparse, start=1)]
        hits = hits[:limit]

        if include_context:
            context_started = time.perf_counter()
            hits = await self._with_context(hits)
            timings["context"] = _since(context_started)
        timings["total"] = _since(started)
        return RetrievalResult(query=query, mode=mode, hits=hits, timings_ms=timings)

    async def _dense(
        self, query: str, depth: int, filters: SearchFilters | None
    ) -> tuple[str, list[ChunkHit], dict[str, float]]:
        started = time.perf_counter()
        vector = await asyncio.to_thread(self._embedder.embed_query, query)
        searching = time.perf_counter()
        hits = await self._searcher.search_dense(vector.tolist(), limit=depth, filters=filters)
        timings = {"embed": _ms(searching - started), "dense_search": _since(searching)}
        return "dense", hits, timings

    async def _sparse(
        self, query: str, depth: int, filters: SearchFilters | None
    ) -> tuple[str, list[ChunkHit], dict[str, float]]:
        started = time.perf_counter()
        vector = await asyncio.to_thread(self._sparse_encoder.encode_query, query)
        searching = time.perf_counter()
        hits = await self._searcher.search_sparse(vector, limit=depth, filters=filters)
        timings = {"encode": _ms(searching - started), "bm25_search": _since(searching)}
        return "bm25", hits, timings

    async def _rerank(self, query: str, hits: list[ChunkHit], limit: int) -> list[ChunkHit]:
        assert self._reranker is not None
        candidates = hits[: max(self._config.rerank_top_k, limit)]
        passages = [index_text(hit.title, hit.section, hit.text) for hit in candidates]
        scores = await asyncio.to_thread(self._reranker.score, query, passages)
        # Stable sort: candidates the reranker scores equally keep their fused order.
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        return [replace(candidates[i], score=float(scores[i])) for i in order]

    async def _with_context(self, hits: list[ChunkHit]) -> list[ChunkHit]:
        assert self._passages is not None
        passages = await self._passages.passages([hit.parent_span for hit in hits])
        return [replace(hit, context=passages.get(hit.parent_span)) for hit in hits]


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _since(started: float) -> float:
    return _ms(time.perf_counter() - started)
