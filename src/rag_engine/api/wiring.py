"""Builds the real retriever on first use, so the API starts without loading any model."""

import asyncio
import logging

from qdrant_client import AsyncQdrantClient

from rag_engine.adapters.postgres_documents import PostgresDocumentStore
from rag_engine.adapters.qdrant_index import QdrantChunkIndex
from rag_engine.core.config import Settings
from rag_engine.core.errors import RetrievalUnavailableError
from rag_engine.core.models import RetrievalMode, RetrievalResult, SearchFilters
from rag_engine.core.ports import Reranker
from rag_engine.indexing.embedders import SentenceTransformerEmbedder
from rag_engine.indexing.sparse import Bm25SparseEncoder
from rag_engine.retrieval.cross_encoder import CrossEncoderReranker
from rag_engine.retrieval.retriever import HybridRetriever, RetrievalConfig

logger = logging.getLogger(__name__)


class LazyRetriever:
    """A `Retriever` that connects to the data services and loads the models at the first query.

    Loading the embedding model takes seconds and about half a gigabyte, and the health endpoints
    should answer whether or not the models are available, so none of that happens at startup. A
    failed start is retried by the next query.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._retriever: HybridRetriever | None = None
        self._store: PostgresDocumentStore | None = None
        self._index: QdrantChunkIndex | None = None

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        filters: SearchFilters | None = None,
        limit: int | None = None,
        include_context: bool = False,
    ) -> RetrievalResult:
        retriever = await self._ready()
        return await retriever.retrieve(
            query, mode=mode, filters=filters, limit=limit, include_context=include_context
        )

    async def aclose(self) -> None:
        if self._index is not None:
            await self._index.close()
        if self._store is not None:
            await self._store.close()

    async def _ready(self) -> HybridRetriever:
        async with self._lock:
            if self._retriever is None:
                try:
                    self._retriever = await self._build()
                except Exception as error:
                    logger.exception("could not start retrieval")
                    raise RetrievalUnavailableError(type(error).__name__) from error
            return self._retriever

    async def _build(self) -> HybridRetriever:
        settings = self._settings
        embedder, sparse_encoder = await asyncio.gather(
            asyncio.to_thread(
                SentenceTransformerEmbedder,
                settings.embedding_model,
                revision=settings.embedding_revision,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
            ),
            asyncio.to_thread(Bm25SparseEncoder, avg_len=settings.bm25_avg_len),
        )
        reranker: Reranker | None = None
        if settings.reranker_model:
            reranker = await asyncio.to_thread(
                CrossEncoderReranker, settings.reranker_model, device=settings.embedding_device
            )
        if embedder.dimension != settings.embedding_dimension:
            raise ValueError("the embedding model does not produce the indexed dimension")
        store = await PostgresDocumentStore.connect(settings.database_url.get_secret_value())
        index = QdrantChunkIndex(
            AsyncQdrantClient(
                url=settings.qdrant_url,
                timeout=settings.qdrant_timeout_seconds,
                prefer_grpc=settings.qdrant_prefer_grpc,
                grpc_port=settings.qdrant_grpc_port,
            ),
            settings.qdrant_collection,
            search_ef=settings.qdrant_hnsw_ef,
        )
        self._store, self._index = store, index
        return HybridRetriever(
            index,
            embedder,
            sparse_encoder,
            passages=store,
            reranker=reranker,
            config=RetrievalConfig(
                dense_top_k=settings.dense_top_k,
                sparse_top_k=settings.sparse_top_k,
                default_limit=settings.fusion_top_k,
                rerank_top_k=settings.rerank_top_k,
                rrf_k=settings.rrf_k,
                dense_weight=settings.rrf_dense_weight,
                sparse_weight=settings.rrf_sparse_weight,
            ),
        )
