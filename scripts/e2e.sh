#!/usr/bin/env bash
# End-to-end check of everything the repository does so far, on the machine it runs on.
#
#   scripts/e2e.sh            quality gates, the stack, search against the real index, and the
#                             answers of our RAG for 3 questions scored by the benchmark's evaluator
#   scripts/e2e.sh --rivals   also the single-attempt grep and the iterative grep agent on the same
#                             3 questions, and the comparison with its fairness checks
#   scripts/e2e.sh --keep     keep the small runs it creates (eval/runs/e2e-*) when it finishes
#
# A step that needs something this machine does not have (the index, an API key, the benchmark
# checkout) is skipped and says what is missing; only a step that runs and fails makes the script
# fail. The model calls are metered and capped: a full run with --rivals costs about 10 cents.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RIVALS=0
KEEP=0
for arg in "$@"; do
    case "$arg" in
        --rivals) RIVALS=1 ;;
        --keep) KEEP=1 ;;
        *) echo "usage: scripts/e2e.sh [--rivals] [--keep]" >&2; exit 2 ;;
    esac
done

JUDGE="claude-haiku-5-5"
RUNS=eval/runs
PASSED=0
FAILED=0
SKIPPED=0
FAILURES=()

step() { printf '\n== %s\n' "$1"; }
pass() { PASSED=$((PASSED + 1)); echo "   PASS  $1"; }
skip() { SKIPPED=$((SKIPPED + 1)); echo "   SKIP  $1"; }
fail() { FAILED=$((FAILED + 1)); FAILURES+=("$1"); echo "   FAIL  $1"; }

# check "name" command...: run it, show its output only when it fails.
check() {
    local name="$1"
    shift
    local log
    log=$(mktemp)
    if "$@" > "$log" 2>&1; then
        pass "$name"
    else
        fail "$name"
        tail -25 "$log" | sed 's/^/         /'
    fi
    rm -f "$log"
}

has_key() { grep -q '^ANTHROPIC_API_KEY=.\+' .env 2> /dev/null; }
has_benchmark() { [ -d data/benchmark/EnterpriseRAG-Bench/generated_data/sources ] || docker volume inspect rag_bench_sources > /dev/null 2>&1; }

# Run an evaluation command inside the image, next to PostgreSQL, with the repository's data.
# The model key is passed by name so it never appears on a command line.
in_stack() {
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    docker compose --profile ingest run --rm --no-deps -T --user "$(id -u):$(id -g)" \
        -v "$PWD/eval:/work/eval" -v "$PWD/data:/work/data:ro" -w /work \
        -e ANTHROPIC_API_KEY -e LOG_LEVEL=WARNING \
        --entrypoint python worker -m rag_engine.evaluation.cli "$@"
}

cleanup() {
    if [ "$KEEP" = 0 ]; then
        rm -rf "$RUNS"/e2e-rag "$RUNS"/e2e-grep "$RUNS"/e2e-agent "$RUNS"/e2e-comparison.*
    fi
}
trap cleanup EXIT
cleanup

step "1. Quality gates"
if command -v uv > /dev/null 2>&1; then
    check "lint" uv run ruff check .
    check "format" uv run ruff format --check .
    check "types" uv run mypy
    check "unit tests" uv run pytest -q
    check "retrieval regression gate and published numbers" \
        uv run python -m rag_engine.evaluation.cli gate
else
    skip "uv is not installed (see docs/setup.md)"
fi

step "2. The stack"
export RERANKER_MODEL="${RERANKER_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}"
check "docker compose up --wait" docker compose up -d --build --wait --wait-timeout 600
# The evaluation commands below run in the worker image: make sure it carries the current code.
check "build the worker image" docker compose --profile ingest build worker

step "3. The index and search"
# The ingestion worker reports how many documents are staged and indexed and how many points the
# vector index holds. An empty stack is fine (the data has not been loaded yet); a half-loaded one
# is not.
index_check() {
    docker compose --profile ingest run --rm --no-deps -T worker status 2> /dev/null | python3 -c '
import json, sys
text = sys.stdin.read()
status = json.loads(text[text.index("{"):])
docs = status["documents"]
if docs["staged"] == 0:
    sys.exit(3)
complete = (
    docs["sparse_indexed"] == docs["dense_indexed"] == docs["staged"]
    and status["points"] == docs["chunks"]
    and status["points_without_dense"] == 0
)
staged, chunks, points = docs["staged"], docs["chunks"], status["points"]
print(f"{staged:,} documents, {chunks:,} chunks, {points:,} points")
sys.exit(0 if complete else 1)
'
}
index_check > /tmp/e2e-index.$$ 2>&1
case $? in
    0) pass "the index is complete ($(cat /tmp/e2e-index.$$))" ;;
    3) skip "the stack has no data loaded (docs/setup.md, step 5)" ;;
    *) fail "the index is incomplete or unreadable"; cat /tmp/e2e-index.$$ ;;
esac
rm -f /tmp/e2e-index.$$

if command -v uv > /dev/null 2>&1; then
    # The index tests skip themselves when the stack has no index loaded.
    log=$(mktemp)
    if uv run pytest -m integration -rs > "$log" 2>&1; then
        pass "integration tests"
        if grep -q "no index loaded" "$log"; then
            skip "search tests: the stack has no index loaded (docs/setup.md, step 5)"
        fi
    else
        fail "integration tests"
        tail -25 "$log" | sed 's/^/         /'
    fi
    rm -f "$log"
else
    skip "uv is not installed"
fi

step "4. Our RAG: answers for 3 questions, scored by the benchmark's evaluator"
if ! has_key; then
    skip "no ANTHROPIC_API_KEY in .env"
elif [ ! -f eval/cache/candidates.jsonl.gz ] || [ ! -f eval/cache/rerank-BAAI_bge-reranker-v2-m3.jsonl.gz ]; then
    skip "no stored candidates in eval/cache (they come from the index: docs/evaluation.md)"
else
    check "answer 3 questions" in_stack answer --run e2e-rag --per-type 1 --limit 3 \
        --rerank eval/cache/rerank-BAAI_bge-reranker-v2-m3.jsonl.gz --budget-usd 0.05 \
        --runs-dir "$RUNS"
    if has_benchmark; then
        check "judge them with the benchmark's evaluator" \
            scripts/benchmark/evaluate.sh "$RUNS/e2e-rag" "$JUDGE" 0.3
    else
        skip "judging: run scripts/benchmark/setup.sh first"
    fi
fi

step "5. The rivals and the comparison"
if [ "$RIVALS" = 0 ]; then
    skip "not asked for (add --rivals)"
elif [ ! -f "$RUNS/e2e-rag/results-$JUDGE.json" ]; then
    skip "needs step 4 to have run and been judged"
else
    check "single-attempt grep on the same questions" \
        scripts/benchmark/run_grep_oneshot.sh "$RUNS/e2e-grep" "$RUNS/e2e-rag" "$JUDGE" 0.3
    check "judge the single-attempt grep" scripts/benchmark/evaluate.sh "$RUNS/e2e-grep" "$JUDGE" 0.3
    mkdir -p "$RUNS/e2e-agent"
    cp "$RUNS/e2e-rag/questions.txt" "$RUNS/e2e-agent/questions.txt"
    check "iterative grep agent on the same questions" \
        scripts/benchmark/run_agent.sh "$RUNS/e2e-agent" "$JUDGE" 0.6 3
    check "judge the agent" env STRIP_CITATIONS=1 \
        scripts/benchmark/evaluate.sh "$RUNS/e2e-agent" "$JUDGE" 0.3
    check "compare the three, with the fairness checks" in_stack compare \
        --arm "Hybrid + rerank (ours)=$RUNS/e2e-rag" \
        --arm "Iterative grep agent=$RUNS/e2e-agent" \
        --arm "Single-attempt grep=$RUNS/e2e-grep" \
        --out-md "$RUNS/e2e-comparison.md" --out-json "$RUNS/e2e-comparison.json"
fi

printf '\n== Result: %d passed, %d failed, %d skipped\n' "$PASSED" "$FAILED" "$SKIPPED"
if [ "$FAILED" -gt 0 ]; then
    printf '   failed: %s\n' "${FAILURES[@]}"
    exit 1
fi
