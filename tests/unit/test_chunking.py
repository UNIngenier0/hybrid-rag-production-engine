import pytest

from fakes import long_text, make_document
from rag_engine.core.models import Chunk
from rag_engine.ingestion.chunking import Chunker, ChunkingConfig
from rag_engine.ingestion.cleaning import ELEMENT_SEPARATOR

CONFIG = ChunkingConfig()


def chunks_of(content: str) -> list[Chunk]:
    return Chunker().chunk(make_document("dsid_1", content))


def test_an_empty_document_has_no_chunks() -> None:
    assert chunks_of("   ") == []


def test_a_short_document_is_one_chunk_spanning_the_text() -> None:
    document = make_document("dsid_1", "A short note about the rollout.")

    [chunk] = Chunker().chunk(document)

    assert chunk.text == document.content
    assert (chunk.char_start, chunk.char_end) == (0, len(document.content))
    assert (chunk.parent_start, chunk.parent_end) == (0, len(document.content))


def test_chunks_reproduce_the_text_and_respect_the_maximum_size() -> None:
    document = make_document("dsid_1", long_text(paragraphs=40))

    chunks = Chunker().chunk(document)

    assert len(chunks) > 5
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert document.content[chunk.char_start : chunk.char_end] == chunk.text
        assert 0 < len(chunk.text) <= CONFIG.max_chars
    words = set(document.content.split())
    assert words == {word for chunk in chunks for word in chunk.text.split()}


def test_most_chunks_are_close_to_the_target_size() -> None:
    chunks = chunks_of(long_text(paragraphs=60))

    sizes = [len(c.text) for c in chunks[:-1]]

    assert sum(size >= CONFIG.target_chars * 0.6 for size in sizes) >= len(sizes) * 0.9


def test_markdown_headings_start_sections_that_label_the_chunks() -> None:
    body_a, body_b = "alpha " * 120, "bravo " * 120
    chunks = chunks_of(f"# Alpha\n\n{body_a}\n\n# Bravo\n\n{body_b}")

    labels = [c.section for c in chunks]

    assert labels[0] == "Alpha"
    assert "Bravo" in labels
    assert all("bravo" not in c.text for c in chunks if c.section == "Alpha")


def test_field_lines_start_sections() -> None:
    content = "description:\n" + "design the thing " * 50 + "\n\ncomments:\n" + "we agree " * 80
    chunks = chunks_of(content)

    assert chunks[0].section == "description"
    assert "comments" in {c.section for c in chunks}


def test_decoded_list_elements_keep_their_own_chunks_and_the_separator_is_dropped() -> None:
    first, second = "x " * 400, "y " * 400
    chunks = chunks_of(f"{first}{ELEMENT_SEPARATOR}{second}")

    assert all("-----" not in c.text for c in chunks)
    assert all(not ("x" in c.text and "y" in c.text) for c in chunks)


def test_text_without_any_line_breaks_is_still_split() -> None:
    # Flattened documents once made the whole text a single block, and every chunk 11k characters.
    flat = " ".join(f"This is sentence number {i} of a long flat document." for i in range(300))

    chunks = chunks_of(flat)

    assert len(chunks) > 5
    assert max(len(c.text) for c in chunks) <= CONFIG.max_chars


def test_text_with_no_spaces_at_all_is_cut_into_overlapping_windows() -> None:
    chunks = chunks_of("x" * 4000)

    assert len(chunks) >= 3
    assert all(len(c.text) <= CONFIG.max_chars for c in chunks)
    assert chunks[1].char_start < chunks[0].char_end  # overlap


def test_a_huge_heading_cannot_become_a_huge_label() -> None:
    chunks = chunks_of("# " + "word " * 300 + "\n\nbody")

    assert all(c.section is None or len(c.section) <= 100 for c in chunks)


def test_parent_spans_contain_their_children_and_group_neighbours() -> None:
    chunks = chunks_of(long_text(paragraphs=60))

    for chunk in chunks:
        assert chunk.parent_start <= chunk.char_start
        assert chunk.char_end <= chunk.parent_end
    parents = {(c.parent_start, c.parent_end) for c in chunks}
    assert 1 < len(parents) < len(chunks)
    assert all(
        end - start <= CONFIG.parent_target_chars + CONFIG.max_chars for start, end in parents
    )


def test_chunking_is_deterministic() -> None:
    document = make_document("dsid_1", long_text(paragraphs=30))

    assert Chunker().chunk(document) == Chunker().chunk(document)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_chars": 0},
        {"target_chars": 2000, "max_chars": 1500},
        {"overlap_chars": 800},
        {"parent_target_chars": 1000},
    ],
)
def test_inconsistent_sizes_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)
