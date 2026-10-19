import hashlib
from pathlib import Path

import pytest

from rag_engine.ingestion import download
from rag_engine.ingestion.download import DatasetFile


def fake_file(content: bytes) -> DatasetFile:
    return DatasetFile(
        "fake", "data/fake.bin", "fake/fake.bin", len(content), hashlib.sha256(content).hexdigest()
    )


def test_a_verified_file_on_disk_is_not_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = fake_file(b"corpus")
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "fake.bin").write_bytes(b"corpus")
    monkeypatch.setattr(download, "_fetch", lambda *args: pytest.fail("should not download"))

    assert download.download(file, tmp_path) == tmp_path / "fake" / "fake.bin"


def test_a_download_is_verified_and_moved_into_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = fake_file(b"corpus")
    monkeypatch.setattr(
        download, "_fetch", lambda url, partial, size: partial.write_bytes(b"corpus")
    )

    path = download.download(file, tmp_path)

    assert path.read_bytes() == b"corpus"
    assert not list(tmp_path.rglob("*.part"))


def test_a_corrupt_download_is_rejected_and_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = fake_file(b"corpus")
    monkeypatch.setattr(
        download, "_fetch", lambda url, partial, size: partial.write_bytes(b"garbage")
    )

    with pytest.raises(ValueError, match="does not match"):
        download.download(file, tmp_path)

    assert not (tmp_path / "fake" / "fake.bin").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_the_pinned_hashes_cover_both_dataset_files() -> None:
    assert set(download.FILES) == {"documents", "questions"}
    assert all(len(file.sha256) == 64 for file in download.FILES.values())
