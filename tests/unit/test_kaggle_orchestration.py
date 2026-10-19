import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "kaggle" / "orchestrate.py"
spec = importlib.util.spec_from_file_location("orchestrate", SCRIPT)
assert spec and spec.loader
orchestrate = importlib.util.module_from_spec(spec)
sys.modules["orchestrate"] = orchestrate
spec.loader.exec_module(orchestrate)


def render(tmp_path: Path, **job_overrides: object) -> Path:
    wheel = tmp_path / "rag_engine-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"PK\x03\x04 pretend wheel \x00\x01\x02" * 50)
    job = orchestrate.job_for("0-1", 5000, None) | job_overrides
    return orchestrate.render(tmp_path / "kernel", "someone", wheel, job)


def test_the_kernel_metadata_asks_for_a_private_gpu_kernel_with_internet(tmp_path: Path) -> None:
    metadata = json.loads((render(tmp_path) / "kernel-metadata.json").read_text())

    assert metadata["id"] == "someone/hybrid-rag-embeddings"
    assert metadata["title"] == "hybrid-rag-embeddings"  # slugifies to the id
    assert metadata["code_file"] == "embed_corpus.py"
    assert (metadata["is_private"], metadata["enable_gpu"], metadata["enable_internet"]) == (
        "true", "true", "true",
    )  # fmt: skip


def test_the_rendered_script_loads_and_embeds_the_exact_wheel(tmp_path: Path) -> None:
    source = (render(tmp_path) / "embed_corpus.py").read_text()
    wheel_bytes = (tmp_path / "rag_engine-0.1.0-py3-none-any.whl").read_bytes()
    namespace: dict[str, object] = {"__name__": "kernel"}  # not "__main__": nothing runs

    exec(compile(source, "embed_corpus.py", "exec"), namespace)

    assert namespace["WHEEL_SHA256"] == hashlib.sha256(wheel_bytes).hexdigest()
    assert base64.b64decode(str(namespace["WHEEL_B64"])) == wheel_bytes


def test_no_placeholder_is_left_behind(tmp_path: Path) -> None:
    source = (render(tmp_path) / "embed_corpus.py").read_text()

    for placeholder in ("__WHEEL_NAME__", "__WHEEL_SHA256__", "__WHEEL_B64__", "__JOB__"):
        assert placeholder not in source


def test_the_job_carries_the_shard_range_and_limits(tmp_path: Path) -> None:
    source = (render(tmp_path, limit_docs=100) / "embed_corpus.py").read_text()
    namespace: dict[str, object] = {"__name__": "kernel"}

    exec(compile(source, "embed_corpus.py", "exec"), namespace)

    assert namespace["JOB"] == {
        "first_shard": 0,
        "last_shard": 1,
        "docs_per_shard": 5000,
        "batch_size": 128,
        "limit_docs": 100,
        "cpu_workers": 0,
    }


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            'someone/hybrid-rag-embeddings has status "KernelWorkerStatus.RUNNING"',
            "KernelWorkerStatus.RUNNING",
        ),
        ('x has status "COMPLETE"', "COMPLETE"),
        ("nothing useful", None),
    ],
)
def test_the_kernel_status_is_read_from_the_cli_output(output: str, expected: str | None) -> None:
    assert orchestrate.parse_status(output) == expected


def test_a_single_shard_job_covers_just_that_shard() -> None:
    job = orchestrate.job_for("3", 5000, None)

    assert (job["first_shard"], job["last_shard"]) == (3, 3)
