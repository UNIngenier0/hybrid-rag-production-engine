#!/usr/bin/env bash
# Score a run with the benchmark's own evaluator, using Claude as the judge.
#
#   scripts/benchmark/evaluate.sh eval/runs/<run> [judge-model] [budget-usd]
#
# The run folder holds answers.jsonl. Results go to <run>/results-<judge-model>.json and what the
# judge cost to <run>/cost-judge-<judge-model>.json. The judge defaults to claude-haiku-5-5 (use
# claude-sonnet-5-5 for a stronger and pricier one) and the budget to 2 dollars.
# Needs scripts/benchmark/setup.sh to have run, Docker, and ANTHROPIC_API_KEY in .env.
#
# STRIP_CITATIONS=1 lets the evaluator remove citations and source lines from the answers first,
# which the benchmark does by default and which suits the agent, whose answers end with sources.
# Our own answers are written without citation markers and do not need it.
#
# Every system is scored against the original gold documents and answers (--no-correction), so
# runs of different systems are comparable. The benchmark's default flow lets three judges
# rewrite the gold set separately for each system, which would make the systems incomparable.
# These numbers use Claude as the judge, not the GPT model of the public leaderboard, and are
# not comparable with it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

run="${1:?usage: evaluate.sh eval/runs/<run> [judge-model] [budget-usd]}"
judge="${2:-claude-haiku-5-5}"
budget="${3:-2}"
[ -f "$ROOT/$run/answers.jsonl" ] || { echo "no answers.jsonl in $run" >&2; exit 1; }
require_benchmark

strip_flag="--skip-citation-stripping"
[ "${STRIP_CITATIONS:-0}" = 1 ] && strip_flag=""

meter_start "$budget"
trap 'meter_stop "$ROOT/$run/cost-judge-$judge.json" || true' EXIT

docker run --rm --network "$NETWORK" "${BENCH_ENV[@]}" -e JUDGE="$judge" -e RUN="$run" -e STRIP_FLAG="$strip_flag" \
    -v "$ROOT":/work -w "/work/$BENCH" -v bench_venv:/venv python:3.12.15-slim bash -c '
set -euo pipefail
[ -x /venv/bin/python ] || python -m venv /venv
/venv/bin/pip install -q -r requirements.txt 2>&1 | grep -v -i "notice" || true
export LLM_MODEL_NAME="$JUDGE" CHEAP_LLM_MODEL_NAME="$JUDGE"
/venv/bin/python -m src.scripts.answer_evaluation.metrics_based_eval \
    --answers-file "/work/$RUN/answers.jsonl" \
    --results-file "/work/$RUN/results-$JUDGE.json" \
    --questions-file questions.jsonl \
    --no-correction $STRIP_FLAG --parallelism 6
'
