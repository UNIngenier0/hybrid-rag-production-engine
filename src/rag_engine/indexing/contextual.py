"""The text that gets embedded and BM25-encoded for a chunk."""

# Bump when the embedded text format changes; it is part of the index fingerprint.
CONTEXT_VERSION = "v1"

# Keeps the header from crowding out the chunk, whatever a title or label holds.
_MAX_HEADER_CHARS = 300


def index_text(title: str, section: str | None, text: str) -> str:
    """Prefix the chunk with its document title and section label.

    A chunk lifted out of its document often loses what it is about: a Slack message does not
    say which channel it was in, and a Jira comment does not repeat the ticket title. Putting the
    title and section in front restores that context for both the dense and the sparse index.
    The stored and displayed text stays the plain chunk text.
    """
    header = title if section is None or section == title else f"{title}\n{section}"
    header = header[:_MAX_HEADER_CHARS]
    return f"{header}\n\n{text}" if header else text
