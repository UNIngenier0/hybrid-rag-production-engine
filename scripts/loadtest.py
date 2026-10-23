"""Load test of the running API: latency and throughput of /v1/search or /v1/query.

    uv run python scripts/loadtest.py --endpoint search --mode hybrid --requests 40 --concurrency 4

Every request asks a different benchmark question, so neither the query cache nor warm disk pages
make a run look better than the next one. Turn the limits off on the server while you measure
(RATE_LIMIT_SEARCH_PER_MINUTE=0), or the limiter does the refusing. Each run is one row of the
report; the numbers describe the machine they ran on, which the report records.
"""

import argparse
import asyncio
import json
import math
import platform
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq

QUESTIONS = Path("data/raw/questions/test.parquet")


def percentile(ordered: list[float], share: float) -> float:
    """Nearest rank: the smallest value that at least `share` of the values do not exceed."""
    if not ordered:
        return float("nan")
    rank = max(1, math.ceil(share * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


async def one(
    client: httpx.AsyncClient, endpoint: str, question: str, options: dict[str, Any]
) -> tuple[int, float, str]:
    body = {"query": question, **options}
    started = time.perf_counter()
    try:
        response = await client.post(f"/v1/{endpoint}", json=body)
        status = response.status_code
        note = ""
        if status == 200 and endpoint == "query":
            note = response.json().get("status", "")
    except httpx.HTTPError as error:
        status, note = 0, type(error).__name__
    return status, time.perf_counter() - started, note


async def run(
    url: str,
    endpoint: str,
    questions: list[str],
    concurrency: int,
    options: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[tuple[int, float, str]] = []

    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=timeout_seconds) as client:

        async def worker(question: str) -> None:
            async with semaphore:
                results.append(await one(client, endpoint, question, options))

        started = time.perf_counter()
        await asyncio.gather(*(worker(question) for question in questions))
        elapsed = time.perf_counter() - started

    good = sorted(seconds for status, seconds, _ in results if status == 200)
    statuses = Counter(status for status, _, _ in results)
    return {
        "endpoint": endpoint,
        "concurrency": concurrency,
        "requests": len(results),
        "ok": statuses[200],
        "statuses": {str(code): count for code, count in sorted(statuses.items())},
        "seconds": round(elapsed, 2),
        "throughput_rps": round(len(results) / elapsed, 3),
        "latency_s": {
            "mean": round(statistics.fmean(good), 3) if good else None,
            "p50": round(percentile(good, 0.50), 3) if good else None,
            "p95": round(percentile(good, 0.95), 3) if good else None,
            "p99": round(percentile(good, 0.99), 3) if good else None,
            "max": round(good[-1], 3) if good else None,
        },
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Load test",
        "",
        f"{report['machine']}. Server settings that matter: {report['server']}.",
        "Every request asks a different benchmark question.",
        "",
        "| Endpoint | Options | Concurrency | Requests | OK | Other statuses | Requests/s "
        "| Mean | p50 | p95 | p99 | Max |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["runs"]:
        other = ", ".join(f"{code}: {n}" for code, n in row["statuses"].items() if code != "200")
        latency = row["latency_s"]
        lines.append(
            f"| {row['endpoint']} | {row['options']} | {row['concurrency']} | {row['requests']} "
            f"| {row['ok']} | {other or '-'} | {row['throughput_rps']} | {latency['mean']} s "
            f"| {latency['p50']} s | {latency['p95']} s | {latency['p99']} s | {latency['max']} s |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--endpoint", choices=["search", "query"], default="search")
    parser.add_argument("--mode", default="hybrid", help="retrieval mode sent with every request")
    parser.add_argument("--limit", type=int, default=10, help="chunks per search")
    parser.add_argument("--verify", action="store_true", help="query: run the citation judge")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 2, 4], help="one run per value"
    )
    parser.add_argument("--warmup", type=int, default=2, help="requests before measuring")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--server", default="", help="settings to note in the report")
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    questions = [row["question"] for row in pq.read_table(args.questions).to_pylist()]
    needed = args.warmup + args.requests * len(args.concurrency)
    if needed > len(questions):
        raise SystemExit(f"{needed} distinct questions are needed, the file has {len(questions)}")
    random.Random(args.seed).shuffle(questions)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    options: dict[str, Any] = {"mode": args.mode}
    if args.endpoint == "search":
        options["limit"] = args.limit
    else:
        options["verify"] = args.verify

    async def everything() -> list[dict[str, Any]]:
        cursor = 0
        if args.warmup:
            await run(
                args.url, args.endpoint, questions[: args.warmup], 1, options, headers, args.timeout
            )
            cursor = args.warmup
        runs = []
        for concurrency in args.concurrency:
            batch = questions[cursor : cursor + args.requests]
            cursor += args.requests
            row = await run(
                args.url, args.endpoint, batch, concurrency, options, headers, args.timeout
            )
            row["options"] = ", ".join(f"{k}={v}" for k, v in options.items())
            print(json.dumps(row))
            runs.append(row)
        return runs

    runs = asyncio.run(everything())
    report = {
        "machine": f"{platform.machine()} {platform.system()}, client on the same host",
        "server": args.server or "defaults",
        "runs": runs,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.write_text(render(report), encoding="utf-8")
    print(render(report))


if __name__ == "__main__":
    main()
