"""Sizing profile: run cleaning and chunking over the corpus and measure what indexing will cost.

Nothing here talks to a service. It answers the questions that decide the indexing plan: how many
chunks the corpus becomes, how long they are, how many tokens will be embedded, and how fast the
CPU-bound stages run.
"""

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rag_engine.indexing.contextual import index_text
from rag_engine.ingestion.chunking import Chunker
from rag_engine.ingestion.parquet_source import CleaningStats, iter_document_batches

_BUCKET = 25  # histogram resolution, in characters


@dataclass(slots=True)
class SourceProfile:
    documents: int = 0
    chars: int = 0
    chunks: int = 0
    single_chunk_documents: int = 0
    parents: int = 0
    chunk_chars: Counter[int] = field(default_factory=Counter)  # bucket -> count
    indexed_chars: int = 0  # characters actually sent to the encoders (title + section + text)

    def percentile(self, fraction: float) -> int:
        """Chunk length at the given quantile, to histogram resolution."""
        threshold = fraction * self.chunks
        seen = 0
        for bucket in sorted(self.chunk_chars):
            seen += self.chunk_chars[bucket]
            if seen >= threshold:
                return (bucket + 1) * _BUCKET
        return 0


@dataclass(slots=True)
class CorpusProfile:
    sources: dict[str, SourceProfile] = field(default_factory=dict)
    cleaning: CleaningStats = field(default_factory=CleaningStats)
    seconds: float = 0.0
    # Every n-th chunk's indexed text, kept so the caller can tokenize a sample.
    sample_texts: list[str] = field(default_factory=list)

    def total(self) -> SourceProfile:
        total = SourceProfile()
        for profile in self.sources.values():
            total.documents += profile.documents
            total.chars += profile.chars
            total.chunks += profile.chunks
            total.single_chunk_documents += profile.single_chunk_documents
            total.parents += profile.parents
            total.chunk_chars.update(profile.chunk_chars)
            total.indexed_chars += profile.indexed_chars
        return total


def profile_corpus(
    path: Path,
    chunker: Chunker,
    *,
    batch_size: int = 2000,
    limit: int | None = None,
    sample_every: int = 100,
) -> CorpusProfile:
    result = CorpusProfile()
    started = time.perf_counter()
    seen = 0
    chunk_number = 0
    for batch in iter_document_batches(path, batch_size=batch_size, stats=result.cleaning):
        for item in batch:
            document = item.document
            profile = result.sources.setdefault(document.source_type, SourceProfile())
            chunks = chunker.chunk(document)
            profile.documents += 1
            profile.chars += len(document.content)
            profile.chunks += len(chunks)
            profile.single_chunk_documents += len(chunks) == 1
            profile.parents += len({(c.parent_start, c.parent_end) for c in chunks})
            for chunk in chunks:
                profile.chunk_chars[len(chunk.text) // _BUCKET] += 1
                text = index_text(document.title, chunk.section, chunk.text)
                profile.indexed_chars += len(text)
                chunk_number += 1
                if chunk_number % sample_every == 0:
                    result.sample_texts.append(text)
            seen += 1
            if limit is not None and seen >= limit:
                result.seconds = time.perf_counter() - started
                return result
    result.seconds = time.perf_counter() - started
    return result
