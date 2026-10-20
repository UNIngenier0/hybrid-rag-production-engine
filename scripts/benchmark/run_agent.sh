#!/usr/bin/env bash
# Run the benchmark's own iterative-grep agent on the questions of a run, with the same model.
#
#   scripts/benchmark/run_agent.sh eval/runs/<run> [model] [budget-usd] [parallelism] [extra args]
#
# The run folder must hold questions.txt (the question IDs, one per line), as written by the
# answer command. The agent searches the raw documents with grep, find, jq and friends, reads the
# ones it picks and writes answers.jsonl (resumable). Its cost is metered and capped: when the
# budget is spent the agent is stopped and what it answered is kept. Defaults: claude-haiku-5-5,
# 5 dollars, 4 parallel questions and 4 CPUs (AGENT_CPUS). The usage report goes to
# <run>/cost-agent.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

run="${1:?usage: run_agent.sh eval/runs/<run> [model] [budget-usd] [parallelism] [extra args]}"
model="${2:-claude-haiku-5-5}"
budget="${3:-5}"
parallelism="${4:-4}"
shift $(( $# < 4 ? $# : 4 ))
[ -f "$ROOT/$run/questions.txt" ] || { echo "no questions.txt in $run" >&2; exit 1; }
require_benchmark

# The agent takes a questions file, so write the run's questions in the benchmark's format.
python3 - "$ROOT/$BENCH/questions.jsonl" "$ROOT/$run/questions.txt" "$ROOT/$run/questions.jsonl" <<'PY'
import json, sys
source, ids_file, out = sys.argv[1:4]
wanted = [line.strip() for line in open(ids_file) if line.strip()]
rows = {}
for line in open(source):
    row = json.loads(line)
    rows[row["question_id"]] = line
with open(out, "w") as handle:
    handle.writelines(rows[i] for i in wanted)
print(f"{len(wanted)} questions for the agent")
PY

# How the answers are produced is recorded next to them, so a comparison can check it.
python3 - "$ROOT/$run/run.json" "$model" "$budget" "$parallelism" <<'PY'
import json, sys
path, model, budget, parallelism = sys.argv[1:5]
manifest = {
    "system": "grep-agent",
    "model": model,
    "passages": None,
    "documents": "reported by the agent",
    "parallelism": int(parallelism),
    "budget_usd": float(budget),
}
open(path, "w").write(json.dumps(manifest, indent=2) + "\n")
PY

meter_start "$budget"
trap 'docker rm -f rag-agent > /dev/null 2>&1 || true; meter_stop "$ROOT/$run/cost-agent.json" || true' EXIT

docker rm -f rag-agent > /dev/null 2>&1 || true
docker run -d --name rag-agent --network "$NETWORK" --cpus "${AGENT_CPUS:-4}" "${BENCH_ENV[@]}" $(sources_mount) \
    -e MODEL="$model" -e RUN="$run" -e PARALLELISM="$parallelism" -e EXTRA="$*" \
    -v "$ROOT":/work -w "/work/$BENCH" -v bench_venv:/venv python:3.12.15-slim bash -c '
set -euo pipefail
apt-get update -qq && apt-get install -y -qq --no-install-recommends jq ripgrep tree > /dev/null
[ -x /venv/bin/python ] || python -m venv /venv
/venv/bin/pip install -q -r requirements.txt 2>&1 | grep -v -i "notice" || true
export LLM_MODEL_NAME="$MODEL" CHEAP_LLM_MODEL_NAME="$MODEL"
# The agent asks two questions before it starts: whether to proceed although a command of its
# allowlist is missing (yes), and whether to rebuild the document index, which ships with the
# repository (no). The budget cap is what protects the spending.
printf "y\nn\n" | /venv/bin/python -u -m src.scripts.answer_generation.agent_retrieval \
    --questions-file "/work/$RUN/questions.jsonl" --output "/work/$RUN/answers.jsonl" \
    --parallelism "$PARALLELISM" --resume $EXTRA
' > /dev/null

echo "agent started; budget \$$budget, model $model"
while docker ps --format '{{.Names}}' | grep -q '^rag-agent$'; do
    sleep 30
    # (the file does not exist until the first answer; pipefail would turn that into an error)
    done_count=$( (cat "$ROOT/$run/answers.jsonl" 2> /dev/null || true) | wc -l)
    spent=$(meter_usage | python3 -c "import json,sys; print(json.load(sys.stdin)['spent_usd'])")
    echo "$(date +%H:%M:%S) answers: $(echo "$done_count" | tr -d ' ')  spent: \$$spent"
    if python3 -c "import sys; sys.exit(0 if float('$spent') >= float('$budget') else 1)"; then
        echo "budget spent: stopping the agent"
        docker kill rag-agent > /dev/null
    fi
done
docker logs rag-agent 2>&1 | tail -15
