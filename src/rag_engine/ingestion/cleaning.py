"""Turn a raw dataset record into a clean `SourceDocument`.

The benchmark corpus has quirks that would hurt retrieval or break storage, all found by
profiling the full corpus:

- ~115k Gmail documents (and a few hundred from other sources) are a serialized list of strings
  (JSON, or the Python `repr` of a list) instead of text, with escaped newlines.
- ~18k documents are a single line whose newlines were written as the two characters `\\n`, which
  hides their headings and paragraphs from the chunker.
- 37 documents contain NUL characters, which PostgreSQL rejects, and more contain other control
  characters.
- ~1k titles are a copy of the whole document body, sometimes spanning many lines.

Cleaning is lossless for the document text apart from control characters, so evaluation against
the benchmark's gold documents is unaffected.
"""

import ast
import hashlib
import json
import re
from dataclasses import dataclass

from rag_engine.core.models import SourceDocument

# Bump when cleaning changes what the text of a document is; it is part of the index fingerprint.
CLEANING_VERSION = "v1"

# Separates the elements of a decoded list (for example the messages of an email thread). The
# chunker splits on it, so it is defined here and shared.
ELEMENT_SEPARATOR = "\n\n-----\n\n"

# Share of real newlines, relative to escaped ones, under which text counts as flattened.
_FLATTENED_RATIO = 0.1

MAX_TITLE_CHARS = 200
DERIVED_TITLE_CHARS = 160

_CONTROL_CHARS = {
    code: " " for code in [*range(0x00, 0x20), 0x7F, 0x2028, 0x2029] if code not in (0x0A, 0x09)
}
_TITLE_PREFIX = re.compile(r"^\s*(?:#+\s*)?(?:title\s*:\s*)?", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CleaningFlags:
    """What cleaning had to repair in one document, for the profiling report."""

    decoded_list: bool = False
    unescaped_newlines: bool = False
    control_chars: bool = False
    title_repaired: bool = False


def normalize_document(
    doc_id: str, source_type: str, title: str | None, content: str | None
) -> tuple[SourceDocument, CleaningFlags]:
    """Clean one record. Never raises on odd input: it repairs what it can and keeps the rest."""
    raw_content = content or ""
    elements = decode_serialized_list(raw_content)
    if elements is not None:
        fixed = [_unescape_newlines(element) for element in elements]
        unescaped = fixed != elements
        text = ELEMENT_SEPARATOR.join(fixed)
    else:
        text = _unescape_newlines(raw_content)
        unescaped = text != raw_content

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = text.translate(_CONTROL_CHARS).strip()
    control_chars = cleaned != text.strip()

    clean_title, title_repaired = _clean_title(title or "", cleaned)
    document = SourceDocument(
        doc_id=doc_id,
        content_sha256=hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
        source_type=source_type,
        title=clean_title,
        content=cleaned,
    )
    flags = CleaningFlags(
        decoded_list=elements is not None,
        unescaped_newlines=unescaped,
        control_chars=control_chars,
        title_repaired=title_repaired,
    )
    return document, flags


def decode_serialized_list(content: str) -> list[str] | None:
    """Decode content that is a JSON list or a Python-repr list of strings, else return None."""
    if not content.startswith("["):
        return None
    try:
        value = json.loads(content)
    except ValueError:
        try:
            value = ast.literal_eval(content)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value
    return None


def _unescape_newlines(text: str) -> str:
    """Turn escaped newlines (the two characters backslash and n) back into real ones.

    Only when the text is clearly flattened: at least two escapes and almost no real newlines.
    Text that already has real line breaks may legitimately contain `\\n` (code, logs), so it is
    left alone.
    """
    escaped = text.count("\\n")
    if escaped >= 2 and text.count("\n") <= escaped * _FLATTENED_RATIO:
        return text.replace("\\n", "\n").replace("\\t", "\t")
    return text


def _clean_title(title: str, content: str) -> tuple[str, bool]:
    """Return the title and whether it had to be repaired."""
    stripped = title.translate(_CONTROL_CHARS).strip()
    if stripped and "\n" not in stripped and len(stripped) <= MAX_TITLE_CHARS:
        cleaned = _SPACES.sub(" ", stripped)
        return cleaned, cleaned != title
    # Missing, multi-line or oversized: the field holds a copy of the body (or nothing), so the
    # title becomes the first non-empty line of whichever text is available.
    lines = (stripped or content).split("\n")
    first_line = next((line for line in lines if line.strip()), "")
    candidate = _SPACES.sub(" ", _TITLE_PREFIX.sub("", first_line)).strip()
    if len(candidate) > DERIVED_TITLE_CHARS:
        candidate = candidate[:DERIVED_TITLE_CHARS].rsplit(" ", 1)[0]
    return candidate, True
