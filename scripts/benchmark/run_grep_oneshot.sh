#!/usr/bin/env bash
# Run the single-attempt grep baseline on the questions of another run, with the same model.
#
#   scripts/benchmark/run_grep_oneshot.sh eval/runs/<new-run> eval/runs/<run-to-match> \
#       [model] [budget-usd]
#
# For each question a model writes one search pattern, ripgrep runs it once over the raw
# documents, and the same answer step as the other systems answers from the five files with the
# most matches. The questions are the ones in <run-to-match>/questions.txt. Cost is metered and
# capped (default claude-haiku-5-5 and 1 dollar). The report goes to <new-run>/cost-grep.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
METER="rag-meter-grep" # its own proxy, so it can run beside the agent

run="${1:?usage: run_grep_oneshot.sh eval/runs/<new-run> eval/runs/<run-to-match> [model] [budget]}"
match="${2:?name the run whose questions.txt to answer}"
model="${3:-claude-haiku-5-5}"
budget="${4:-1}"
[ -f "$ROOT/$match/questions.txt" ] || { echo "no questions.txt in $match" >&2; exit 1; }
require_benchmark

if docker volume inspect "$SOURCES_VOLUME" > /dev/null 2>&1; then
    corpus_mount=(-v "$SOURCES_VOLUME:/corpus:ro")
else
    corpus_mount=(-v "$ROOT/$BENCH/generated_data:/corpus:ro")
fi

meter_start "$budget"
trap 'meter_stop "$ROOT/$run/cost-grep.json" || true' EXIT
mkdir -p "$ROOT/$run"

docker run --rm --network "$NETWORK" --cpus "${GREP_CPUS:-2}" --env-file "$ROOT/.env" \
    -e ANTHROPIC_API_KEY=via-the-metering-proxy -e "ANTHROPIC_BASE_URL=http://$METER:8080" \
    -e DATABASE_URL=postgresql://unused:unused@unused/unused -e LOG_LEVEL=WARNING \
    -e PYTHONPATH=/work/src -e MODEL="$model" -e RUN="$run" -e MATCH="$match" -e BUDGET="$budget" \
    "${corpus_mount[@]}" -v "$ROOT":/work -w /work -v rag_grep_venv:/venv python:3.12.15-slim bash -c '
set -euo pipefail
apt-get update -qq && apt-get install -y -qq --no-install-recommends ripgrep > /dev/null
[ -x /venv/bin/python ] || python -m venv /venv
/venv/bin/pip install -q numpy httpx pyarrow pydantic pydantic-settings 2>&1 | grep -v -i notice || true
/venv/bin/python -u -m rag_engine.evaluation.cli grep-answer --run "$(basename "$RUN")" \
    --runs-dir "$(dirname "$RUN")" --questions-from "/work/$MATCH/questions.txt" \
    --corpus /corpus/sources --model "$MODEL" --budget-usd "$BUDGET" --concurrency 3
'
