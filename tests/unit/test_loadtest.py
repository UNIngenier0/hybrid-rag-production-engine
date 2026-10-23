import importlib.util
import math
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "loadtest.py"
spec = importlib.util.spec_from_file_location("loadtest", SCRIPT)
assert spec and spec.loader
loadtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loadtest)


def test_percentiles_use_the_nearest_rank() -> None:
    values = [float(n) for n in range(1, 101)]

    assert loadtest.percentile(values, 0.50) == 50.0
    assert loadtest.percentile(values, 0.95) == 95.0
    assert loadtest.percentile(values, 0.99) == 99.0


def test_a_few_values_still_give_a_percentile_inside_the_range() -> None:
    assert loadtest.percentile([3.0], 0.99) == 3.0
    assert loadtest.percentile([1.0, 2.0, 3.0], 0.95) == 3.0
    assert loadtest.percentile([1.0, 2.0, 3.0], 0.50) == 2.0


def test_no_values_give_no_percentile() -> None:
    assert math.isnan(loadtest.percentile([], 0.5))


def test_the_report_has_one_row_per_run() -> None:
    row = {
        "endpoint": "search",
        "options": "mode=hybrid",
        "concurrency": 2,
        "requests": 10,
        "ok": 9,
        "statuses": {"200": 9, "429": 1},
        "throughput_rps": 1.5,
        "latency_s": {"mean": 1.0, "p50": 0.9, "p95": 2.0, "p99": 2.5, "max": 3.0},
    }

    text = loadtest.render({"machine": "arm64 Linux", "server": "defaults", "runs": [row]})

    assert "| search | mode=hybrid | 2 | 10 | 9 | 429: 1 | 1.5 |" in text
    assert text.count("\n| search") == 1
