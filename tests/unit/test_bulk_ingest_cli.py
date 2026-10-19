import argparse

import pytest

from rag_engine.workers import bulk_ingest


def args(limit: int | None, allow: bool = False) -> argparse.Namespace:
    return argparse.Namespace(limit=limit, allow_slow_device=allow)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_the_whole_corpus_is_refused_on_a_cpu_or_laptop_gpu(device: str) -> None:
    with pytest.raises(SystemExit, match="refusing to embed"):
        bulk_ingest._refuse_slow_device(device, args(limit=None))
    with pytest.raises(SystemExit, match="refusing to embed"):
        bulk_ingest._refuse_slow_device(device, args(limit=10_000))


def test_a_small_run_is_allowed_anywhere() -> None:
    bulk_ingest._refuse_slow_device("cpu", args(limit=100))


def test_a_cuda_gpu_is_always_allowed() -> None:
    bulk_ingest._refuse_slow_device("cuda:0", args(limit=None))


def test_the_flag_overrides_the_refusal() -> None:
    bulk_ingest._refuse_slow_device("cpu", args(limit=None, allow=True))


def test_shard_arguments_must_be_a_valid_fraction() -> None:
    assert bulk_ingest._shard("1/4") == (1, 4)
    for bad in ("4/4", "-1/4", "1/0"):
        with pytest.raises(argparse.ArgumentTypeError):
            bulk_ingest._shard(bad)


def test_commands_parse() -> None:
    parser = bulk_ingest._parser()

    assert parser.parse_args(["index", "dense", "--limit", "5"]).limit == 5
    assert parser.parse_args(["load-vectors", "--dir", "somewhere"]).dir is not None
    assert parser.parse_args(["stage"]).command == "stage"
