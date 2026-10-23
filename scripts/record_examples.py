"""Record real /v1/query responses for the project's static demo (the GitHub Pages site).

    uv run python scripts/record_examples.py --ids qst_0001 qst_0411 --reranker <model name>

Each question is sent to the running service with the cache off and the citation check on, and
the whole response is kept, untouched, in site/examples.json. The page replays these; nothing in
them is written by hand. Re-run it to refresh the recording after a change to the pipeline.
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq

QUESTIONS = Path("data/raw/questions/test.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--ids", nargs="+", required=True, help="benchmark question IDs")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--reranker", required=True, help="the reranker the server runs, to note")
    parser.add_argument("--out", type=Path, default=Path("site/examples.json"))
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    by_id = {row["question_id"]: row for row in pq.read_table(args.questions).to_pylist()}
    missing = [question_id for question_id in args.ids if question_id not in by_id]
    if missing:
        raise SystemExit(f"not in the benchmark questions: {', '.join(missing)}")
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    examples = []
    with httpx.Client(base_url=args.url, headers=headers, timeout=args.timeout) as client:
        for question_id in args.ids:
            row = by_id[question_id]
            started = time.perf_counter()
            response = client.post(
                "/v1/query", json={"query": row["question"], "verify": True, "cache": False}
            )
            response.raise_for_status()
            print(f"{question_id}: {response.status_code} in {time.perf_counter() - started:.0f} s")
            examples.append(
                {
                    "id": question_id,
                    "type": row["question_type"],
                    "question": row["question"],
                    "response": response.json(),
                }
            )
    recording = {
        "recorded_on": datetime.date.today().isoformat(),
        "server": {"reranker": args.reranker, "hardware": "a laptop CPU, no GPU"},
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(recording, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
