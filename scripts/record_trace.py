"""Record the real retrieval trace of one question for the project's static site.

    uv run python scripts/record_trace.py --id qst_0411 --reranker <model name>

The question goes to the running service as one hybrid + rerank search that asks for the whole
candidate pool. Every hit says where the dense search, BM25 and the fusion ranked it and what the
reranker scored it, which is all the site's walkthrough draws. The chunk text is not kept: the
walkthrough shows titles and positions, and the answer it leads to is in site/examples.json.
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq

QUESTIONS = Path("data/raw/questions/test.parquet")
KEPT = (
    "rank",
    "score",
    "dense_rank",
    "bm25_rank",
    "fusion_rank",
    "chunk_id",
    "doc_id",
    "source_type",
    "title",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--id", required=True, help="benchmark question ID")
    parser.add_argument("--pool", type=int, default=30, help="the service's rerank pool size")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--reranker", required=True, help="the reranker the server runs, to note")
    parser.add_argument("--out", type=Path, default=Path("site/trace.json"))
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    by_id = {row["question_id"]: row for row in pq.read_table(args.questions).to_pylist()}
    if args.id not in by_id:
        raise SystemExit(f"not in the benchmark questions: {args.id}")
    row = by_id[args.id]
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    started = time.perf_counter()
    with httpx.Client(base_url=args.url, headers=headers, timeout=args.timeout) as client:
        response = client.post(
            "/v1/search",
            json={"query": row["question"], "mode": "hybrid+rerank", "limit": args.pool},
        )
        response.raise_for_status()
    body = response.json()
    print(f"{args.id}: {len(body['hits'])} hits in {time.perf_counter() - started:.0f} s")
    if len(body["hits"]) != args.pool:
        raise SystemExit(
            f"expected the {args.pool} candidates of the pool, got {len(body['hits'])}"
        )

    trace = {
        "recorded_on": datetime.date.today().isoformat(),
        "question_id": args.id,
        "query": row["question"],
        "mode": body["mode"],
        "reranker": args.reranker,
        "timings_ms": body["timings_ms"],
        "pool": [{key: hit[key] for key in KEPT} for hit in body["hits"]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
