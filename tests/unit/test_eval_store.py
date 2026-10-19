from pathlib import Path

from fakes import make_hit
from rag_engine.evaluation.store import append_jsonl, hit_from_dict, hit_to_dict, read_jsonl


def test_rows_appended_in_separate_calls_read_back_as_one_stream(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl.gz"

    append_jsonl(path, [{"n": 1}, {"n": 2}])
    append_jsonl(path, [{"n": 3}])

    assert [row["n"] for row in read_jsonl(path)] == [1, 2, 3]


def test_a_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert list(read_jsonl(tmp_path / "absent.jsonl.gz")) == []


def test_non_ascii_text_survives(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl.gz"

    append_jsonl(path, [{"text": "café ✓ 日本語"}])

    assert next(read_jsonl(path))["text"] == "café ✓ 日本語"


def test_a_hit_survives_the_round_trip() -> None:
    hit = make_hit("a", doc_id="d1", section="Intro", dense_rank=3, fusion_rank=2, score=0.5)

    assert hit_from_dict(hit_to_dict(hit)) == hit
