import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fakes import make_hit
from rag_engine.evaluation import cli
from rag_engine.evaluation.store import append_jsonl, hit_to_dict


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    questions = tmp_path / "questions.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "question_id": "q1",
                    "question_type": "basic",
                    "question": "what?",
                    "expected_doc_ids": "['doc-a']",
                }
            ]
        ),
        questions,
    )
    candidates = tmp_path / "candidates.jsonl.gz"
    hits = [hit_to_dict(make_hit("a", doc_id="doc-a")), hit_to_dict(make_hit("b"))]
    append_jsonl(candidates, [{"question_id": "q1", "dense": hits, "bm25": hits[::-1]}])
    return questions, candidates


def test_report_writes_the_markdown_and_the_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    questions, candidates = write_inputs(tmp_path)
    out_md, out_json = tmp_path / "out" / "ablation.md", tmp_path / "out" / "ablation.json"

    cli.main(
        [
            "report",
            "--questions", str(questions),
            "--candidates", str(candidates),
            "--out-md", str(out_md),
            "--out-json", str(out_json),
        ]
    )  # fmt: skip

    assert "# Retrieval ablation" in out_md.read_text(encoding="utf-8")
    assert json.loads(out_json.read_text(encoding="utf-8"))["questions"] == 1
    assert "Retrieval ablation" in capsys.readouterr().out


def test_a_rerank_argument_without_a_file_is_refused(tmp_path: Path) -> None:
    questions, candidates = write_inputs(tmp_path)

    with pytest.raises(SystemExit):
        cli.main(
            [
                "report",
                "--questions", str(questions),
                "--candidates", str(candidates),
                "--rerank", "bge",
                "--out-md", str(tmp_path / "a.md"),
                "--out-json", str(tmp_path / "a.json"),
            ]
        )  # fmt: skip


def test_every_command_is_listed_in_the_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])

    out = capsys.readouterr().out
    assert all(
        command in out for command in ("collect", "rerank", "export-pool", "answer", "report")
    )


def test_questions_without_gold_documents_are_reranked_when_they_have_candidates() -> None:
    from rag_engine.evaluation.questions import Question

    questions = [
        Question("q1", "basic", "a?", ("doc",), (), "", ()),
        Question("q2", "info_not_found", "b?", (), (), "", ()),
        Question("q3", "info_not_found", "c?", (), (), "", ()),
    ]
    candidates = {"q1": object(), "q2": object()}  # q3 was never searched

    assert [q.question_id for q in cli._to_rerank(questions, candidates, None, None)] == [
        "q1",
        "q2",
    ]
    assert [
        q.question_id for q in cli._to_rerank(questions, candidates, ["info_not_found"], None)
    ] == ["q2"]
    assert [q.question_id for q in cli._to_rerank(questions, candidates, None, 1)] == ["q1"]
