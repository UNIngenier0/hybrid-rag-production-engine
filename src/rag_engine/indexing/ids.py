"""Deterministic identifiers for index points."""

import uuid

# Fixed namespace: changing it would re-key every point, so it must never change.
_NAMESPACE = uuid.UUID("5d0f6f1e-3c1a-4c55-9a43-6a8f0e7b2c11")


def chunk_point_id(doc_id: str, content_sha256: str, chunk_index: int) -> str:
    """ID of a chunk's point, derived from what the chunk is.

    Re-indexing the same document therefore overwrites its points instead of adding new ones.
    The content hash is part of the key because a `doc_id` is not unique in the benchmark data,
    and because a changed document must not be mistaken for the old one.
    """
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{content_sha256}:{chunk_index}"))
