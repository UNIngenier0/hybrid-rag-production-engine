"""Stream the benchmark's Parquet documents without loading the file into memory."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from rag_engine.core.models import SequencedDocument
from rag_engine.ingestion.cleaning import CleaningFlags, normalize_document

_COLUMNS = ["doc_id", "source_type", "title", "content"]


@dataclass(slots=True)
class CleaningStats:
    """Counts of what cleaning repaired, accumulated over a run."""

    documents: int = 0
    decoded_lists: int = 0
    unescaped_newlines: int = 0
    control_chars: int = 0
    titles_repaired: int = 0
    by_source: dict[str, int] = field(default_factory=dict)

    def add(self, source_type: str, flags: CleaningFlags) -> None:
        self.documents += 1
        self.decoded_lists += flags.decoded_list
        self.unescaped_newlines += flags.unescaped_newlines
        self.control_chars += flags.control_chars
        self.titles_repaired += flags.title_repaired
        self.by_source[source_type] = self.by_source.get(source_type, 0) + 1


def count_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def iter_document_batches(
    path: Path,
    *,
    batch_size: int = 2000,
    start_seq: int = 0,
    stats: CleaningStats | None = None,
) -> Iterator[list[SequencedDocument]]:
    """Yield cleaned documents in batches, in file order.

    `seq` is the zero-based row number in the file. Rows before `start_seq` are skipped without
    being cleaned, which is how a restarted run resumes. The benchmark file is a single row
    group, so the reader still scans from the start, but that costs seconds, not minutes.
    """
    reader = pq.ParquetFile(path)
    seq = 0
    for batch in reader.iter_batches(batch_size=batch_size, columns=_COLUMNS):
        batch_start, seq = seq, seq + batch.num_rows
        if seq <= start_seq:
            continue
        columns = [batch.column(name).to_pylist() for name in _COLUMNS]
        batch_documents: list[SequencedDocument] = []
        for offset, (doc_id, source_type, title, content) in enumerate(zip(*columns, strict=True)):
            row_seq = batch_start + offset
            if row_seq < start_seq:
                continue
            document, flags = normalize_document(doc_id, source_type, title, content)
            if stats is not None:
                stats.add(source_type, flags)
            batch_documents.append(SequencedDocument(seq=row_seq, document=document))
        if batch_documents:
            yield batch_documents
