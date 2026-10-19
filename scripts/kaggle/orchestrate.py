"""Run the GPU embedding job on Kaggle: render the kernel, push it, wait for it, fetch the shards.

    python scripts/kaggle/orchestrate.py run --username <kaggle-user> --wheel dist/<wheel> \\
        --shards 0-1 --label smoke

Authentication is the Kaggle CLI's: set KAGGLE_API_TOKEN (or save the token in
~/.kaggle/access_token). Standard library only; it shells out to the `kaggle` CLI, which is in the
`kaggle` dependency group (`uv run --group kaggle python scripts/kaggle/orchestrate.py ...`).
Build the wheel from the commit being run first: `uv build --wheel`.
"""

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).parent
SLUG = "hybrid-rag-embeddings"
# Kaggle's P100 is not supported by recent PyTorch builds; the T4 is, and often comes as a pair.
ACCELERATOR = "NvidiaTeslaT4"
_STATUS = re.compile(r'has status "(?P<status>[^"]+)"')
_FAILED = ("ERROR", "CANCEL")


def render(
    out_dir: Path, username: str, wheel: Path, job: dict[str, object], slug: str = SLUG
) -> Path:
    """Write the kernel script (with the wheel embedded) and its metadata under `out_dir`."""
    wheel_bytes = wheel.read_bytes()
    template = (HERE / "kernel_template.py").read_text(encoding="utf-8")
    script = (
        template.replace("__WHEEL_NAME__", wheel.name)
        .replace("__WHEEL_SHA256__", hashlib.sha256(wheel_bytes).hexdigest())
        .replace("__WHEEL_B64__", base64.encodebytes(wheel_bytes).decode("ascii"))
        .replace("__JOB__", json.dumps(job))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "embed_corpus.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": "embed_corpus.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def job_for(
    shards: str, docs_per_shard: int, limit_docs: int | None, *, cpu_workers: int = 0
) -> dict[str, object]:
    first, _, last = shards.partition("-")
    return {
        "first_shard": int(first),
        "last_shard": int(last) if last else int(first),
        "docs_per_shard": docs_per_shard,
        "batch_size": 128,
        "limit_docs": limit_docs,
        "cpu_workers": cpu_workers,  # only for testing the script away from a GPU
    }


def push_with_retries(folder: Path, accelerator: str, attempts: int = 4) -> None:
    """Push a kernel, retrying the transient server errors the Kaggle API sometimes returns."""
    for attempt in range(1, attempts + 1):
        try:
            kaggle("kernels", "push", "-p", str(folder), "--accelerator", accelerator)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(
                f"push failed (attempt {attempt}/{attempts}); retrying", file=sys.stderr, flush=True
            )
            time.sleep(20 * attempt)


def kaggle(*arguments: str, capture: bool = False) -> str:
    result = subprocess.run(["kaggle", *arguments], check=True, text=True, capture_output=capture)
    return result.stdout if capture else ""


def parse_status(output: str) -> str | None:
    match = _STATUS.search(output)
    return match.group("status") if match else None


def wait(kernel: str, *, poll_seconds: int = 60, timeout_seconds: int = 13 * 3600) -> str:
    """Block until the kernel stops running. Returns the final status."""
    deadline = time.time() + timeout_seconds
    last = None
    failures = 0
    while time.time() < deadline:
        try:
            status = parse_status(kaggle("kernels", "status", kernel, capture=True)) or "UNKNOWN"
            failures = 0
        except subprocess.CalledProcessError:
            # The Kaggle API fails now and then for a moment; a long wait should survive that.
            failures += 1
            if failures >= 10:
                raise
            time.sleep(poll_seconds)
            continue
        if status != last:
            print(f"[{time.strftime('%H:%M:%S')}] {kernel}: {status}", flush=True)
            last = status
        if status.upper().endswith("COMPLETE") or any(word in status.upper() for word in _FAILED):
            return status
        time.sleep(poll_seconds)
    raise TimeoutError(f"{kernel} was still running after {timeout_seconds} s")


def fetch(kernel: str, output_dir: Path, vectors_dir: Path) -> list[Path]:
    """Download the kernel output and move its shards into the folder `load-vectors` reads."""
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel, "-p", str(output_dir), "-o")
    shards = sorted(output_dir.rglob("vectors-*.parquet"))
    if not shards:
        raise SystemExit(f"the kernel produced no shards; see the logs in {output_dir}")
    vectors_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        shutil.move(str(shard), vectors_dir / shard.name)
    return [vectors_dir / shard.name for shard in shards]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["run", "render", "quota"])
    parser.add_argument("--username", help="your Kaggle username")
    parser.add_argument("--wheel", type=Path, help="wheel built with `uv build --wheel`")
    parser.add_argument("--shards", default="0-0", help="first-last shard to embed, e.g. 0-1")
    parser.add_argument("--docs-per-shard", type=int, default=5000)
    parser.add_argument("--limit-docs", type=int, help="stop after N documents (smoke test)")
    parser.add_argument("--label", default="run", help="names the working folder under data/kaggle")
    parser.add_argument("--vectors-dir", type=Path, default=Path("data/vectors"))
    parser.add_argument("--accelerator", default=ACCELERATOR, help="Kaggle accelerator name")
    parser.add_argument(
        "--slug", default=SLUG, help="kernel name; use a different one to run two jobs at once"
    )
    parser.add_argument(
        "--cpu-workers", type=int, default=0, help="testing only: allow this many CPU workers"
    )
    args = parser.parse_args(argv)

    if args.command == "quota":
        kaggle("quota")
        return 0
    if not args.username or not args.wheel:
        parser.error("--username and --wheel are required")
    job = job_for(args.shards, args.docs_per_shard, args.limit_docs, cpu_workers=args.cpu_workers)
    work = Path("data/kaggle") / args.label
    render(work / "kernel", args.username, args.wheel, job, args.slug)
    print(f"kernel rendered in {work / 'kernel'} for job {job}")
    if args.command == "render":
        return 0

    kernel = f"{args.username}/{args.slug}"
    kaggle("quota")
    push_with_retries(work / "kernel", args.accelerator)
    status = wait(kernel)
    try:
        kaggle("kernels", "logs", kernel)
    except subprocess.CalledProcessError:
        print("could not read the kernel logs", file=sys.stderr)
    if not status.upper().endswith("COMPLETE"):
        print(f"the kernel ended with status {status}", file=sys.stderr)
        return 1
    shards = fetch(kernel, work / "output", args.vectors_dir)
    print(f"{len(shards)} shards in {args.vectors_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
