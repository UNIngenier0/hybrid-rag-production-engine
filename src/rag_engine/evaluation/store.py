"""Gzipped JSON-lines files: appendable, so a long run can be interrupted and continued."""

import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_engine.core.models import ChunkHit


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Add rows to the file. Each call writes its own gzip member; readers see one stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row: dict[str, Any] = json.loads(line)
                yield row


def hit_to_dict(hit: ChunkHit) -> dict[str, Any]:
    return asdict(hit)


def hit_from_dict(row: dict[str, Any]) -> ChunkHit:
    return ChunkHit(**row)
