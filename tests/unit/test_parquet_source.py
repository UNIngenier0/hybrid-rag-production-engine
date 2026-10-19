import json
from pathlib import Path

from fakes import write_corpus
from rag_engine.ingestion.parquet_source import CleaningStats, count_rows, iter_document_batches


def corpus(tmp_path: Path) -> Path:
    rows = [(f"dsid_{i}", "slack", f"chan-{i}", f"message number {i}") for i in range(7)]
    rows[2] = ("dsid_2", "gmail", "mail", json.dumps(["From: a\nhi", "From: b\nhey"]))
    return write_corpus(tmp_path / "docs.parquet", rows)


def test_documents_come_in_file_order_with_their_row_numbers(tmp_path: Path) -> None:
    batches = list(iter_document_batches(corpus(tmp_path), batch_size=3))

    assert [[d.seq for d in batch] for batch in batches] == [[0, 1, 2], [3, 4, 5], [6]]
    assert batches[0][1].document.doc_id == "dsid_1"


def test_starting_at_a_row_skips_the_earlier_ones_even_inside_a_batch(tmp_path: Path) -> None:
    batches = list(iter_document_batches(corpus(tmp_path), batch_size=3, start_seq=4))

    assert [d.seq for batch in batches for d in batch] == [4, 5, 6]


def test_nothing_is_yielded_after_the_last_row(tmp_path: Path) -> None:
    assert list(iter_document_batches(corpus(tmp_path), batch_size=3, start_seq=7)) == []


def test_cleaning_stats_count_only_the_rows_that_were_cleaned(tmp_path: Path) -> None:
    stats = CleaningStats()

    list(iter_document_batches(corpus(tmp_path), batch_size=3, stats=stats))

    assert stats.documents == 7
    assert stats.decoded_lists == 1
    assert stats.by_source == {"slack": 6, "gmail": 1}


def test_row_count_comes_from_the_file_metadata(tmp_path: Path) -> None:
    assert count_rows(corpus(tmp_path)) == 7
