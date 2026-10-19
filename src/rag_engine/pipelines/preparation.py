"""Turn documents into the chunks to index: IDs, payloads and the text to encode (CPU-bound)."""

from dataclasses import dataclass, field

from rag_engine.core.models import SequencedDocument
from rag_engine.indexing.contextual import index_text
from rag_engine.indexing.ids import chunk_point_id
from rag_engine.ingestion.chunking import CHUNKER_VERSION, Chunker


@dataclass(slots=True)
class PreparedBatch:
    """The chunks of a batch of documents, before any vector is computed.

    The lists are parallel: entry `i` of each one describes the same chunk.
    """

    last_seq: int
    chunk_counts: dict[int, int]
    seqs: list[int] = field(default_factory=list)
    chunk_indexes: list[int] = field(default_factory=list)
    point_ids: list[str] = field(default_factory=list)
    payloads: list[dict[str, object]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)


def prepare_batch(chunker: Chunker, documents: list[SequencedDocument]) -> PreparedBatch:
    """Chunk documents and build, per chunk, its ID, payload and the text to encode."""
    batch = PreparedBatch(last_seq=documents[-1].seq, chunk_counts={})
    for item in documents:
        document = item.document
        chunks = chunker.chunk(document)
        batch.chunk_counts[item.seq] = len(chunks)
        for chunk in chunks:
            batch.seqs.append(item.seq)
            batch.chunk_indexes.append(chunk.index)
            batch.point_ids.append(chunk_point_id(chunk.doc_id, chunk.content_sha256, chunk.index))
            batch.texts.append(index_text(document.title, chunk.section, chunk.text))
            batch.payloads.append(
                {
                    "doc_id": chunk.doc_id,
                    "content_sha256": chunk.content_sha256,
                    "source_type": document.source_type,
                    "title": document.title,
                    "section": chunk.section,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "parent_start": chunk.parent_start,
                    "parent_end": chunk.parent_end,
                    "chunker_version": CHUNKER_VERSION,
                }
            )
    return batch
