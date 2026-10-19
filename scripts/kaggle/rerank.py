"""Score the retrieval candidates with several cross-encoders on a free Kaggle GPU.

    uv run --group kaggle python scripts/kaggle/rerank.py run --username <kaggle-user> \\
        --wheel dist/<wheel> --candidates eval/cache/pool.jsonl.gz \\
        --questions data/raw/questions/test.parquet

The candidates file to ship is the compact one from `python -m rag_engine.evaluation.cli
export-pool`: the full file is four times larger and uploads slowly.

It uploads the candidates as a private dataset, pushes a kernel that scores them with each model
and downloads one `rerank-<model>.jsonl.gz` per model into `eval/cache`. Authentication is the
Kaggle CLI's, as for the embedding job (see orchestrate.py).
"""

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from orchestrate import kaggle, push_with_retries, wait

HERE = Path(__file__).parent
KERNEL_SLUG = "hybrid-rag-rerank"
DATASET_SLUG = "hybrid-rag-rerank-input"
CANDIDATES_FILE = "candidates.dat"
ACCELERATOR = "NvidiaTeslaT4"
# Small and fast ones are candidates for the CPU of the serving machine; the large ones show how
# much a stronger model would add.
MODELS = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "BAAI/bge-reranker-v2-m3",
)


def prepare_dataset(
    folder: Path, username: str, candidates: Path, questions: Path, slug: str = DATASET_SLUG
) -> Path:
    """Lay out the files of the private dataset and its metadata."""
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.iterdir():
        stale.unlink()
    # Kaggle unpacks anything that looks like an archive (.gz, .zip), so the gzip file travels
    # under a neutral extension and is read back as gzip by the kernel.
    shutil.copyfile(candidates, folder / CANDIDATES_FILE)
    shutil.copyfile(questions, folder / "questions.parquet")
    metadata = {"title": slug, "id": f"{username}/{slug}", "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return folder


def upload_dataset(folder: Path, username: str, slug: str = DATASET_SLUG) -> None:
    """Create the dataset, or add a version when it already exists, and wait until it is ready."""
    reference = f"{username}/{slug}"
    try:
        kaggle("datasets", "status", reference, capture=True)
        kaggle("datasets", "version", "-p", str(folder), "-m", "candidates", "--dir-mode", "skip")
    except subprocess.CalledProcessError:
        kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "skip")
    wait_for_dataset(reference)


def wait_for_dataset(reference: str, timeout_seconds: int = 20 * 60) -> None:
    """Block until Kaggle has processed the dataset (right after creation the call can fail)."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            status = kaggle("datasets", "status", reference, capture=True).strip().lower()
        except subprocess.CalledProcessError:
            status = "not visible yet"
        print(f"dataset {reference}: {status}", flush=True)
        if status.endswith("ready"):
            return
        time.sleep(15)
    raise TimeoutError(f"{reference} was not ready after {timeout_seconds // 60} minutes")


def render(
    out_dir: Path,
    username: str,
    wheel: Path,
    job: dict[str, object],
    *,
    kernel_slug: str = KERNEL_SLUG,
    dataset_slug: str = DATASET_SLUG,
) -> Path:
    wheel_bytes = wheel.read_bytes()
    template = (HERE / "rerank_template.py").read_text(encoding="utf-8")
    script = (
        template.replace("__WHEEL_NAME__", wheel.name)
        .replace("__WHEEL_SHA256__", hashlib.sha256(wheel_bytes).hexdigest())
        .replace("__WHEEL_B64__", base64.encodebytes(wheel_bytes).decode("ascii"))
        .replace("__JOB__", json.dumps(job))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rerank.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": "rerank.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [f"{username}/{dataset_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def fetch(kernel: str, output_dir: Path, destination: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel, "-p", str(output_dir), "-o")
    files = sorted(output_dir.rglob("rerank-*.jsonl.gz"))
    if not files:
        raise SystemExit(f"the kernel produced no scores; see the logs in {output_dir}")
    destination.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.move(str(file), destination / file.name)
    return [destination / file.name for file in files]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["run", "render"])
    parser.add_argument("--username", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=Path("eval/cache/pool.jsonl.gz"))
    parser.add_argument("--questions", type=Path, default=Path("data/raw/questions/test.parquet"))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--pool", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--work", type=Path, default=Path("data/kaggle/rerank"))
    parser.add_argument("--out", type=Path, default=Path("eval/cache"))
    parser.add_argument(
        "--reuse-dataset", action="store_true", help="skip the upload: the dataset is already there"
    )
    args = parser.parse_args(argv)

    job = {"models": args.models, "pool": args.pool, "batch_size": args.batch_size}
    render(args.work / "kernel", args.username, args.wheel, job)
    print(f"kernel rendered in {args.work / 'kernel'} for {len(args.models)} models")
    if args.command == "render":
        return 0

    if args.reuse_dataset:
        wait_for_dataset(f"{args.username}/{DATASET_SLUG}")
    else:
        dataset = prepare_dataset(
            args.work / "dataset", args.username, args.candidates, args.questions
        )
        upload_dataset(dataset, args.username)
    kernel = f"{args.username}/{KERNEL_SLUG}"
    kaggle("quota")
    push_with_retries(args.work / "kernel", ACCELERATOR)
    status = wait(kernel, poll_seconds=30)
    try:
        kaggle("kernels", "logs", kernel)
    except subprocess.CalledProcessError:
        print("could not read the kernel logs", file=sys.stderr)
    if not status.upper().endswith("COMPLETE"):
        print(f"the kernel ended with status {status}", file=sys.stderr)
        return 1
    files = fetch(kernel, args.work / "output", args.out)
    print(f"{len(files)} scores files in {args.out}: {[f.name for f in files]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
