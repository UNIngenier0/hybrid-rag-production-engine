"""The Qdrant adapter's search, run against qdrant-client's in-memory engine (no server)."""

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from fakes import DIMENSION
from rag_engine.adapters.qdrant_index import QdrantChunkIndex
from rag_engine.core.models import ChunkPoint, SearchFilters, SparseVector

pytestmark = [
    pytest.mark.filterwarnings("ignore:Payload indexes have no effect"),
    pytest.mark.filterwarnings("ignore:Local mode performs exact"),
]


def point(
    point_id: str, doc_id: str, source_type: str, terms: list[int], *, title: str = "Title"
) -> ChunkPoint:
    return ChunkPoint(
        point_id=point_id,
        payload={
            "doc_id": doc_id,
            "content_sha256": f"sha-{doc_id}",
            "source_type": source_type,
            "title": title,
            "section": None,
            "chunk_index": 0,
            "text": f"text of {point_id}",
            "char_start": 0,
            "char_end": 20,
            "parent_start": 0,
            "parent_end": 100,
        },
        sparse=SparseVector(indices=terms, values=[1.0] * len(terms)),
    )


def unit_vector(axis: int) -> list[float]:
    return [1.0 if i == axis else 0.0 for i in range(DIMENSION)]


# Qdrant point IDs must be UUIDs.
A, B, C, D = (f"00000000-0000-0000-0000-00000000000{n}" for n in range(1, 5))


@pytest.fixture
async def index() -> AsyncIterator[QdrantChunkIndex]:
    chunk_index = QdrantChunkIndex(AsyncQdrantClient(":memory:"), "test")
    await chunk_index.ensure_ready(DIMENSION)
    await chunk_index.upsert_sparse(
        [
            point(A, "d1", "slack", [1, 2]),
            point(B, "d2", "jira", [2, 3]),
            point(C, "d3", "slack", [3, 4]),
            point(D, "d1", "slack", [9]),
        ]
    )
    await chunk_index.add_dense(
        {A: unit_vector(0), B: unit_vector(1), C: unit_vector(2), D: unit_vector(3)}
    )
    yield chunk_index
    await chunk_index.close()


async def test_dense_search_ranks_by_similarity_and_returns_the_payload(
    index: QdrantChunkIndex,
) -> None:
    query = [0.0] * DIMENSION
    query[1], query[2] = 1.0, 0.5

    hits = await index.search_dense(query, limit=3)

    assert [hit.point_id for hit in hits[:2]] == [B, C]
    assert hits[0].score > hits[1].score > 0
    best = hits[0]
    assert (best.doc_id, best.source_type, best.title) == ("d2", "jira", "Title")
    assert (best.text, best.section, best.chunk_index) == (f"text of {B}", None, 0)
    assert (best.parent_start, best.parent_end) == (0, 100)
    assert best.content_sha256 == "sha-d2"


async def test_dense_search_accepts_a_search_effort(index: QdrantChunkIndex) -> None:
    tuned = QdrantChunkIndex(index._client, "test", search_ef=16)

    hits = await tuned.search_dense(unit_vector(0), limit=2)

    assert hits[0].point_id == A


async def test_sparse_search_matches_on_shared_terms(index: QdrantChunkIndex) -> None:
    hits = await index.search_sparse(SparseVector(indices=[4], values=[1.0]), limit=5)

    assert [hit.point_id for hit in hits] == [C]


async def test_sparse_search_ranks_chunks_by_overlap(index: QdrantChunkIndex) -> None:
    hits = await index.search_sparse(SparseVector(indices=[2, 3], values=[1.0, 1.0]), limit=5)

    assert hits[0].point_id == B
    assert {hit.point_id for hit in hits} == {A, B, C}


async def test_limit_caps_the_number_of_hits(index: QdrantChunkIndex) -> None:
    assert len(await index.search_dense(unit_vector(0), limit=2)) == 2


async def test_source_type_filter_applies_to_both_searches(index: QdrantChunkIndex) -> None:
    filters = SearchFilters(source_types=("slack",))
    query = [1.0] * DIMENSION

    dense = await index.search_dense(query, limit=10, filters=filters)
    sparse = await index.search_sparse(
        SparseVector(indices=[2, 3], values=[1.0, 1.0]), limit=10, filters=filters
    )

    assert {hit.source_type for hit in dense} == {"slack"}
    assert {hit.point_id for hit in dense} == {A, C, D}
    assert {hit.point_id for hit in sparse} == {A, C}


async def test_doc_id_filter_restricts_to_those_documents(index: QdrantChunkIndex) -> None:
    hits = await index.search_dense(
        [1.0] * DIMENSION, limit=10, filters=SearchFilters(doc_ids=("d1", "d3"))
    )

    assert {hit.doc_id for hit in hits} == {"d1", "d3"}
    assert {hit.point_id for hit in hits} == {A, C, D}


async def test_filters_combine_with_and(index: QdrantChunkIndex) -> None:
    hits = await index.search_dense(
        [1.0] * DIMENSION,
        limit=10,
        filters=SearchFilters(source_types=("slack",), doc_ids=("d2", "d3")),
    )

    assert [hit.point_id for hit in hits] == [C]


async def test_empty_filters_do_not_restrict(index: QdrantChunkIndex) -> None:
    unrestricted = await index.search_dense([1.0] * DIMENSION, limit=10)
    empty = await index.search_dense([1.0] * DIMENSION, limit=10, filters=SearchFilters())

    assert len(unrestricted) == len(empty) == 4
