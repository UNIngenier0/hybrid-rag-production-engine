"""Download the EnterpriseRAG-Bench Parquet files, resumable and verified by SHA-256.

Standard library only, so it also runs in a bare notebook environment. The expected hashes are
the ones Hugging Face publishes for the files; if the dataset is ever updated, verification fails
loudly instead of silently mixing corpus versions.
"""

import argparse
import hashlib
import logging
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

_BASE_URL = "https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench/resolve/main"
_CHUNK_BYTES = 1 << 20


class DatasetFile(NamedTuple):
    name: str
    remote_path: str
    local_path: str
    size: int
    sha256: str


DOCUMENTS = DatasetFile(
    name="documents",
    remote_path="data/documents/test.parquet",
    local_path="documents/test.parquet",
    size=1_409_893_131,
    sha256="6b0747bf160af9427b12101537d53056ac592ada9831c1a98ae01fa50a8d2a9f",
)
QUESTIONS = DatasetFile(
    name="questions",
    remote_path="data/questions/test.parquet",
    local_path="questions/test.parquet",
    size=408_737,
    sha256="e25066f4eff3843dd0f3df0d1348113471e072e75007ffe390a0aa83f2a80af2",
)
FILES = {file.name: file for file in (DOCUMENTS, QUESTIONS)}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def download(file: DatasetFile, root: Path, *, attempts: int = 5) -> Path:
    """Fetch one file under `root`, resuming a partial download, and verify its hash."""
    target = root / file.local_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == file.size and sha256_of(target) == file.sha256:
        logger.info("already downloaded", extra={"file": file.name})
        return target

    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            _fetch(f"{_BASE_URL}/{file.remote_path}", partial, file.size)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            logger.warning(
                "download interrupted",
                extra={"file": file.name, "attempt": attempt, "error": str(error)},
            )
            if attempt == attempts:
                raise
            time.sleep(min(5 * attempt, 30))

    actual = sha256_of(partial)
    if actual != file.sha256:
        partial.unlink()
        raise ValueError(f"{file.name}: SHA-256 {actual} does not match the expected {file.sha256}")
    partial.replace(target)
    logger.info("downloaded", extra={"file": file.name, "bytes": file.size})
    return target


def _fetch(url: str, partial: Path, size: int) -> None:
    have = partial.stat().st_size if partial.exists() else 0
    if have > size:
        partial.unlink()
        have = 0
    if have == size:
        return
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        # A server that ignores Range answers 200 with the whole file: start over in that case.
        resumed = have > 0 and response.status == 206
        with partial.open("ab" if resumed else "wb") as out:
            while block := response.read(_CHUNK_BYTES):
                out.write(block)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the EnterpriseRAG-Bench Parquet files")
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    parser.add_argument("files", nargs="*", choices=list(FILES), help="default: all files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    for name in args.files or list(FILES):
        print(download(FILES[name], args.dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
