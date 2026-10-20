#!/usr/bin/env bash
# Fetch the benchmark's own evaluation code and its raw documents into data/benchmark.
#
# The benchmark ships an official evaluator (correctness, completeness, document recall, invalid
# extra documents) and an agent that searches the corpus with grep. Both read the corpus as JSON
# files, which live in the benchmark's Git repository (about 4 GB). A partial, sparse checkout
# fetches only the folders needed. Safe to run again.
#
#   scripts/benchmark/setup.sh            fetch the code and the documents
#   scripts/benchmark/setup.sh --volume   also copy the documents into a Docker volume
#
# The volume is for the grep agent: on macOS and Windows a bind mount reads half a million small
# files far more slowly than a Docker volume does. The copy takes several minutes.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bench="$root/data/benchmark/EnterpriseRAG-Bench"

if [ ! -d "$bench/.git" ]; then
    mkdir -p "$root/data/benchmark"
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git "$bench"
fi
git -C "$bench" sparse-checkout set src answer_evaluation generated_data
git -C "$bench" checkout --quiet

count=$(find "$bench/generated_data/sources" -type f -name '*.json' | wc -l | tr -d ' ')
echo "benchmark ready: $count documents in $bench/generated_data/sources"

if [ "${1:-}" = "--volume" ]; then
    docker volume create rag_bench_sources > /dev/null
    docker run --rm -v "$bench/generated_data":/from:ro -v rag_bench_sources:/to \
        python:3.12.15-slim sh -c 'cp -r /from/. /to/'
    echo "documents copied into the Docker volume rag_bench_sources"
fi
