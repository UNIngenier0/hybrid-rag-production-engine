"""Qdrant adapter: one collection holding a dense and a BM25 sparse vector per chunk."""

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence

from grpc.aio import AioRpcError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_engine.core.errors import RetrievalUnavailableError
from rag_engine.core.models import ChunkHit, ChunkPoint, SearchFilters, SparseVector

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

# What a search raises when Qdrant is down, too slow (the gRPC deadline) or overloaded.
_UNAVAILABLE = (AioRpcError, ResponseHandlingException, UnexpectedResponse, OSError, TimeoutError)


class QdrantChunkIndex:
    """Chunk index. Points are written by the sparse pass and completed by the dense pass.

    Storage is sized for a corpus that does not fit in RAM: dense vectors live on disk in half
    precision with an int8 copy kept in memory for the first-stage search (originals are used to
    rescore), payloads are on disk, and the BM25 index is on disk. IDF is computed by Qdrant, so
    term statistics follow the collection as it grows.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        *,
        upload_batch_size: int = 500,
        indexing_threads: int = 2,
        search_ef: int | None = None,
    ) -> None:
        self._client = client
        self._collection = collection
        self._upload_batch_size = upload_batch_size
        # Index building is CPU-heavy; a small cap keeps a laptop usable while it runs.
        self._indexing_threads = indexing_threads
        # Candidates the dense graph search keeps while it walks; Qdrant's default is the build-time
        # value (128). Lower is faster, mostly when the graph is not in memory, and costs recall.
        self._search_ef = search_ef

    async def close(self) -> None:
        await self._client.close()

    async def ensure_ready(self, dimension: int) -> None:
        if not await self._client.collection_exists(self._collection):
            try:
                await self._create(dimension)
            except Exception:
                # Parallel workers may race to create it; only a missing collection is an error.
                if not await self._client.collection_exists(self._collection):
                    raise
        await self._ensure_payload_indexes()

    async def _create(self, dimension: int) -> None:
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    datatype=models.Datatype.FLOAT16,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True), modifier=models.Modifier.IDF
                )
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            ),
            hnsw_config=models.HnswConfigDiff(max_indexing_threads=self._indexing_threads),
            optimizers_config=models.OptimizersConfigDiff(
                max_optimization_threads=self._indexing_threads
            ),
            on_disk_payload=True,
        )
        logger.info("created collection", extra={"collection": self._collection})

    async def _ensure_payload_indexes(self) -> None:
        for field in ("doc_id", "source_type"):
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    async def upsert_sparse(self, points: Sequence[ChunkPoint]) -> None:
        for start in range(0, len(points), self._upload_batch_size):
            batch = points[start : start + self._upload_batch_size]
            await self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=point.point_id,
                        vector={
                            SPARSE_VECTOR: models.SparseVector(
                                indices=point.sparse.indices, values=point.sparse.values
                            )
                        },
                        payload=point.payload,
                    )
                    for point in batch
                ],
                wait=True,
            )

    async def add_dense(self, vectors: Mapping[str, Sequence[float]]) -> None:
        items = list(vectors.items())
        for start in range(0, len(items), self._upload_batch_size):
            batch = items[start : start + self._upload_batch_size]
            await self._client.update_vectors(
                collection_name=self._collection,
                points=[
                    models.PointVectors(id=point_id, vector={DENSE_VECTOR: list(vector)})
                    for point_id, vector in batch
                ],
                wait=True,
            )

    async def search_dense(
        self, vector: Sequence[float], *, limit: int, filters: SearchFilters | None = None
    ) -> list[ChunkHit]:
        params = models.SearchParams(hnsw_ef=self._search_ef) if self._search_ef else None
        return await self._search(list(vector), DENSE_VECTOR, limit, filters, params)

    async def search_sparse(
        self, vector: SparseVector, *, limit: int, filters: SearchFilters | None = None
    ) -> list[ChunkHit]:
        query = models.SparseVector(indices=vector.indices, values=vector.values)
        return await self._search(query, SPARSE_VECTOR, limit, filters, None)

    async def _search(
        self,
        query: list[float] | models.SparseVector,
        using: str,
        limit: int,
        filters: SearchFilters | None,
        params: models.SearchParams | None,
    ) -> list[ChunkHit]:
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query,
                using=using,
                query_filter=_filter(filters),
                search_params=params,
                limit=limit,
                with_payload=True,
            )
        except _UNAVAILABLE as error:
            # Under load a cold search can outlast the deadline: that is "try again later" for the
            # caller (a 503), not a crash.
            logger.warning("qdrant search failed", exc_info=True)
            raise RetrievalUnavailableError(f"qdrant: {type(error).__name__}") from error
        return [_hit(point) for point in response.points]

    async def count(self) -> int:
        if not await self._client.collection_exists(self._collection):
            return 0
        result = await self._client.count(collection_name=self._collection, exact=True)
        return result.count

    async def count_missing_dense(self) -> int:
        if not await self._client.collection_exists(self._collection):
            return 0
        result = await self._client.count(
            collection_name=self._collection,
            count_filter=models.Filter(
                must_not=[models.HasVectorCondition(has_vector=DENSE_VECTOR)]
            ),
            exact=True,
        )
        return result.count

    async def wait_until_idle(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            info = await self._client.get_collection(self._collection)
            if info.status == models.CollectionStatus.GREEN:
                return
            await asyncio.sleep(2)
        logger.warning("index still optimizing after the wait", extra={"seconds": timeout_seconds})

    async def set_dense_graph(self, enabled: bool, *, m: int = 16, ef_construct: int = 128) -> None:
        """Turn the dense HNSW graph off while vectors are bulk-loaded, then on to build it once.

        Building the graph incrementally while millions of vectors stream in wastes most of the
        indexing work; with `m=0` Qdrant stores the vectors and builds the graph only when it is
        enabled again.
        """
        await self._client.update_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParamsDiff(
                    hnsw_config=models.HnswConfigDiff(
                        m=m if enabled else 0, ef_construct=ef_construct
                    )
                )
            },
        )


def _filter(filters: SearchFilters | None) -> models.Filter | None:
    if filters is None:
        return None
    conditions: list[models.Condition] = []
    for field, values in (("source_type", filters.source_types), ("doc_id", filters.doc_ids)):
        if values:
            conditions.append(
                models.FieldCondition(key=field, match=models.MatchAny(any=list(values)))
            )
    return models.Filter(must=conditions) if conditions else None


def _hit(point: models.ScoredPoint) -> ChunkHit:
    payload = point.payload or {}
    return ChunkHit(
        point_id=str(point.id),
        doc_id=payload["doc_id"],
        content_sha256=payload["content_sha256"],
        source_type=payload["source_type"],
        title=payload["title"],
        section=payload.get("section"),
        chunk_index=payload["chunk_index"],
        text=payload["text"],
        char_start=payload["char_start"],
        char_end=payload["char_end"],
        parent_start=payload["parent_start"],
        parent_end=payload["parent_end"],
        score=point.score,
    )
