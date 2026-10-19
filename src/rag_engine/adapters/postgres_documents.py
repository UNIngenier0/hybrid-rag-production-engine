"""PostgreSQL store for cleaned documents and per-document ingestion progress."""

from collections.abc import Mapping, Sequence
from typing import Self

import asyncpg

from rag_engine.core.models import (
    DocumentProgress,
    PassageSpan,
    SequencedDocument,
    SourceDocument,
)
from rag_engine.core.ports import IndexStage, Shard

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        seq            BIGINT PRIMARY KEY,
        doc_id         TEXT   NOT NULL,
        content_sha256 TEXT   NOT NULL,
        source_type    TEXT   NOT NULL,
        title          TEXT   NOT NULL,
        content        TEXT   NOT NULL,
        n_chars        INTEGER NOT NULL,
        UNIQUE (doc_id, content_sha256)
    )
    """,
    "CREATE INDEX IF NOT EXISTS documents_doc_id_idx ON documents (doc_id)",
    # One row per document once any pass has run. `sparse_at` and `dense_at` are the checkpoints:
    # a pass writes them only after the index has acknowledged the batch.
    """
    CREATE TABLE IF NOT EXISTS index_progress (
        seq       BIGINT PRIMARY KEY REFERENCES documents (seq) ON DELETE CASCADE,
        n_chunks  INTEGER NOT NULL,
        sparse_at TIMESTAMPTZ,
        dense_at  TIMESTAMPTZ
    )
    """,
)

_INSERT_DOCUMENTS = """
    INSERT INTO documents (seq, doc_id, content_sha256, source_type, title, content, n_chars)
    SELECT * FROM unnest(
        $1::bigint[], $2::text[], $3::text[], $4::text[], $5::text[], $6::text[], $7::int[]
    )
    ON CONFLICT DO NOTHING
"""

# What "still needs this pass" means. Dense vectors are attached to points the sparse pass
# created, so a document is only eligible for the dense pass once the sparse pass is done.
_PENDING_PREDICATE: dict[IndexStage, str] = {
    "sparse": "p.sparse_at IS NULL",
    "dense": "p.sparse_at IS NOT NULL AND p.dense_at IS NULL",
}

_PENDING = """
    SELECT d.seq, d.doc_id, d.content_sha256, d.source_type, d.title, d.content
    FROM documents d LEFT JOIN index_progress p ON p.seq = d.seq
    WHERE d.seq > $1 AND d.seq % $3 = $2 AND ({predicate})
    ORDER BY d.seq
    LIMIT $4
"""

# Replacing a point drops its dense vector, so redoing the sparse pass invalidates the dense one.
_MARK_SPARSE = """
    INSERT INTO index_progress (seq, n_chunks, sparse_at)
    SELECT seq, n_chunks, now() FROM unnest($1::bigint[], $2::int[]) AS t(seq, n_chunks)
    ON CONFLICT (seq) DO UPDATE
    SET n_chunks = EXCLUDED.n_chunks, sparse_at = EXCLUDED.sparse_at, dense_at = NULL
"""

_MARK_DENSE = """
    UPDATE index_progress p SET dense_at = now()
    FROM unnest($1::bigint[]) AS t(seq) WHERE p.seq = t.seq
"""

# The text of many spans in one round trip. `substr` counts characters, like the offsets the
# chunker produced, and the (doc_id, content_sha256) key is unique, so each span matches one row.
_PASSAGES = """
    SELECT s.doc_id, s.content_sha256, s.span_start, s.span_end,
           substr(d.content, s.span_start + 1, s.span_end - s.span_start) AS text
    FROM unnest($1::text[], $2::text[], $3::int[], $4::int[])
         AS s(doc_id, content_sha256, span_start, span_end)
    JOIN documents d ON d.doc_id = s.doc_id AND d.content_sha256 = s.content_sha256
"""


class PostgresDocumentStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str, *, max_size: int = 4) -> Self:
        return cls(await asyncpg.create_pool(dsn, min_size=1, max_size=max_size))

    async def close(self) -> None:
        await self._pool.close()

    async def ensure_schema(self) -> None:
        async with self._pool.acquire() as connection:
            for statement in _SCHEMA:
                await connection.execute(statement)

    async def last_staged_seq(self) -> int:
        value = await self._pool.fetchval("SELECT max(seq) FROM documents")
        return -1 if value is None else int(value)

    async def add_documents(self, documents: Sequence[SequencedDocument]) -> int:
        if not documents:
            return 0
        status = await self._pool.execute(
            _INSERT_DOCUMENTS,
            [d.seq for d in documents],
            [d.document.doc_id for d in documents],
            [d.document.content_sha256 for d in documents],
            [d.document.source_type for d in documents],
            [d.document.title for d in documents],
            [d.document.content for d in documents],
            [len(d.document.content) for d in documents],
        )
        return int(status.rsplit(" ", 1)[-1])  # "INSERT 0 <rows>"

    async def pending(
        self, stage: IndexStage, *, after_seq: int, limit: int, shard: Shard | None = None
    ) -> list[SequencedDocument]:
        index, total = shard or (0, 1)
        query = _PENDING.format(predicate=_PENDING_PREDICATE[stage])
        rows = await self._pool.fetch(query, after_seq, index, total, limit)
        return [
            SequencedDocument(
                seq=row["seq"],
                document=SourceDocument(
                    doc_id=row["doc_id"],
                    content_sha256=row["content_sha256"],
                    source_type=row["source_type"],
                    title=row["title"],
                    content=row["content"],
                ),
            )
            for row in rows
        ]

    async def mark_indexed(self, stage: IndexStage, chunk_counts: Mapping[int, int]) -> None:
        if not chunk_counts:
            return
        seqs = list(chunk_counts)
        if stage == "sparse":
            await self._pool.execute(_MARK_SPARSE, seqs, [chunk_counts[s] for s in seqs])
        else:
            await self._pool.execute(_MARK_DENSE, seqs)

    async def chunk_counts(self, seq_start: int, seq_end: int) -> dict[int, int]:
        rows = await self._pool.fetch(
            "SELECT seq, n_chunks FROM index_progress "
            "WHERE seq >= $1 AND seq < $2 AND sparse_at IS NOT NULL",
            seq_start,
            seq_end,
        )
        return {int(row["seq"]): int(row["n_chunks"]) for row in rows}

    async def progress(self) -> DocumentProgress:
        staged = await self._pool.fetchval("SELECT count(*) FROM documents")
        row = await self._pool.fetchrow(
            """
            SELECT count(*) FILTER (WHERE sparse_at IS NOT NULL) AS sparse,
                   count(*) FILTER (WHERE dense_at IS NOT NULL) AS dense,
                   coalesce(sum(n_chunks), 0) AS chunks
            FROM index_progress
            """
        )
        assert row is not None
        return DocumentProgress(
            staged=int(staged),
            sparse_indexed=int(row["sparse"]),
            dense_indexed=int(row["dense"]),
            chunks=int(row["chunks"]),
        )

    async def passages(self, spans: Sequence[PassageSpan]) -> dict[PassageSpan, str]:
        unique = list(dict.fromkeys(spans))
        if not unique:
            return {}
        rows = await self._pool.fetch(
            _PASSAGES,
            [span.doc_id for span in unique],
            [span.content_sha256 for span in unique],
            [span.start for span in unique],
            [span.end for span in unique],
        )
        return {
            PassageSpan(row["doc_id"], row["content_sha256"], row["span_start"], row["span_end"]): (
                row["text"]
            )
            for row in rows
        }
