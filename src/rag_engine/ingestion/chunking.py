"""Structure-aware parent-child chunking.

Documents are split along the structure they already have, from coarse to fine:

1. sections: Markdown headings, `field_name:` lines (Jira, Linear, HubSpot, ...) and the elements
   of a decoded list (the messages of an email thread);
2. paragraphs: blocks separated by blank lines;
3. lines, then sentences, then fixed windows with overlap, only for a block that is still larger
   than `max_chars`.

Consecutive pieces are packed into **child chunks** of about `target_chars`, preferring to end a
chunk at a section boundary. Children are then grouped into **parent** passages of about
`parent_target_chars`. Children are what gets embedded and searched (precise); the parent span is
what the LLM reads (enough context), and it is stored as offsets, never as a second copy of text.

Everything is deterministic: the same document and configuration always give the same chunks,
which is what makes the point IDs, and therefore re-ingestion, idempotent.
"""

import re
from dataclasses import dataclass
from typing import NamedTuple

from rag_engine.core.models import Chunk, SourceDocument
from rag_engine.ingestion.cleaning import ELEMENT_SEPARATOR

CHUNKER_VERSION = "v1"

# A Markdown heading, or a bare `field_name:` line, starts a new section.
_SECTION_MARKER = re.compile(
    r"^(?:#{1,6}[ \t]+(?P<heading>[^\n]+?)|(?P<field>[a-z][a-z0-9_]{0,40}):)[ \t]*$", re.MULTILINE
)
# A label is shown in front of every chunk, so it must stay short whatever the heading holds.
_MAX_LABEL_CHARS = 100
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Sizes are in characters: model-agnostic and cheap. About 3.7 characters make a token."""

    target_chars: int = 1000
    max_chars: int = 1500
    # A chunk is closed at a section boundary once it holds at least this share of the target.
    boundary_fill: float = 0.6
    # A trailing fragment shorter than this is merged into the previous chunk when it fits.
    min_chars: int = 150
    overlap_chars: int = 120
    parent_target_chars: int = 3200

    def __post_init__(self) -> None:
        if not 0 < self.min_chars < self.target_chars <= self.max_chars:
            raise ValueError("expected 0 < min_chars < target_chars <= max_chars")
        if not 0 <= self.overlap_chars < self.max_chars // 2:
            raise ValueError("overlap_chars must be smaller than half of max_chars")
        if not self.max_chars <= self.parent_target_chars:
            raise ValueError("parent_target_chars must be at least max_chars")


class _Span(NamedTuple):
    start: int
    end: int
    section: int  # index into the section list


class _Section(NamedTuple):
    start: int
    end: int
    label: str | None


class Chunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []
        sections = _sections(text)
        pieces = self._pieces(text, sections)
        children = self._pack(pieces)
        parents = self._group_parents(children)
        return [
            Chunk(
                doc_id=document.doc_id,
                content_sha256=document.content_sha256,
                index=index,
                text=text[child.start : child.end],
                char_start=child.start,
                char_end=child.end,
                parent_start=parents[index][0],
                parent_end=parents[index][1],
                section=sections[child.section].label,
            )
            for index, child in enumerate(children)
        ]

    def _pieces(self, text: str, sections: list[_Section]) -> list[_Span]:
        pieces: list[_Span] = []
        for number, section in enumerate(sections):
            for start, end in _paragraphs(text, section.start, section.end):
                pieces.extend(
                    _Span(a, b, number) for a, b in self._split_oversized(text, start, end)
                )
        return pieces

    def _split_oversized(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Break a block larger than `max_chars` along lines, then sentences, then windows."""
        if end - start <= self.config.max_chars:
            return [(start, end)]
        for split in (_lines, _sentences):
            units = split(text, start, end)
            if len(units) > 1:
                result: list[tuple[int, int]] = []
                for unit_start, unit_end in units:
                    result.extend(self._split_oversized(text, unit_start, unit_end))
                return result
        return _windows(text, start, end, self.config.max_chars, self.config.overlap_chars)

    def _pack(self, pieces: list[_Span]) -> list[_Span]:
        config = self.config
        boundary_size = config.target_chars * config.boundary_fill
        children: list[_Span] = []
        current: _Span | None = None
        for piece in pieces:
            if current is None:
                current = piece
                continue
            size = current.end - current.start
            merged_size = piece.end - current.start
            crosses_section = piece.section != current.section
            close = (
                merged_size > config.max_chars
                or (merged_size > config.target_chars and size >= config.min_chars)
                or (crosses_section and size >= boundary_size)
            )
            if close:
                children.append(current)
                current = piece
            else:
                current = _Span(current.start, max(current.end, piece.end), current.section)
        if current is not None:
            children.append(current)
        return self._merge_short_tail(children)

    def _merge_short_tail(self, children: list[_Span]) -> list[_Span]:
        if len(children) >= 2:
            last, previous = children[-1], children[-2]
            if (
                last.end - last.start < self.config.min_chars
                and last.end - previous.start <= self.config.max_chars
            ):
                children[-2:] = [_Span(previous.start, last.end, previous.section)]
        return children

    def _group_parents(self, children: list[_Span]) -> list[tuple[int, int]]:
        """Return, for each child, the span of the parent passage it belongs to."""
        target = self.config.parent_target_chars
        boundary_size = target * self.config.boundary_fill
        groups: list[list[_Span]] = []
        for child in children:
            if groups:
                group = groups[-1]
                size = group[-1].end - group[0].start
                fits = child.end - group[0].start <= target
                ends_at_section = child.section != group[-1].section and size >= boundary_size
                if fits and not ends_at_section:
                    group.append(child)
                    continue
            groups.append([child])
        spans: list[tuple[int, int]] = []
        for group in groups:
            parent = (group[0].start, group[-1].end)
            spans.extend(parent for _ in group)
        return spans


def _sections(text: str) -> list[_Section]:
    """Split a document into contiguous sections, without the separators between elements."""
    sections: list[_Section] = []
    for start, end in _elements(text):
        markers = list(_SECTION_MARKER.finditer(text, start, end))
        if not markers or markers[0].start() > start:
            first_end = markers[0].start() if markers else end
            sections.append(_Section(start, first_end, None))
        for number, marker in enumerate(markers):
            stop = markers[number + 1].start() if number + 1 < len(markers) else end
            label = marker.group("heading") or marker.group("field")
            sections.append(
                _Section(marker.start(), stop, label.strip()[:_MAX_LABEL_CHARS] if label else None)
            )
    return [section for section in sections if text[section.start : section.end].strip()]


def _elements(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    while True:
        found = text.find(ELEMENT_SEPARATOR, position)
        if found == -1:
            spans.append((position, len(text)))
            return spans
        spans.append((position, found))
        position = found + len(ELEMENT_SEPARATOR)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraphs(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = start
    for match in _PARAGRAPH_BREAK.finditer(text, start, end):
        spans.append(_trim(text, position, match.start()))
        position = match.end()
    spans.append(_trim(text, position, end))
    return [(a, b) for a, b in spans if b > a]


def _lines(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = start
    while position < end:
        newline = text.find("\n", position, end)
        stop = end if newline == -1 else newline
        spans.append(_trim(text, position, stop))
        position = stop + 1
    return [(a, b) for a, b in spans if b > a]


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = start
    for match in _SENTENCE_END.finditer(text, start, end):
        spans.append(_trim(text, position, match.start()))
        position = match.end()
    spans.append(_trim(text, position, end))
    return [(a, b) for a, b in spans if b > a]


def _windows(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Fixed windows with overlap, ending at a space when there is one near the window end."""
    spans: list[tuple[int, int]] = []
    position = start
    while position < end:
        stop = min(position + size, end)
        if stop < end:
            space = text.rfind(" ", position + size // 2, stop)
            if space != -1:
                stop = space
        spans.append(_trim(text, position, stop))
        if stop >= end:
            break
        position = max(stop - overlap, position + 1)
    return [(a, b) for a, b in spans if b > a]
