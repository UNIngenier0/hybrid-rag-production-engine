# Shared helpers for the benchmark scripts. Source this file, do not run it.
#
# The benchmark's own tools call the Anthropic API with no cost report and no cap, so they are
# pointed at a metering proxy (meter_proxy.py) that holds the real key, adds up what every call
# costs and refuses calls once the budget is spent.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="data/benchmark/EnterpriseRAG-Bench"
NETWORK="rag-bench-net"
METER="rag-meter"

require_benchmark() {
    [ -d "$ROOT/$BENCH/generated_data/sources" ] || {
        echo "run scripts/benchmark/setup.sh first" >&2
        exit 1
    }
}

# meter_start BUDGET_USD
meter_start() {
    docker network inspect "$NETWORK" > /dev/null 2>&1 || docker network create "$NETWORK" > /dev/null
    docker rm -f "$METER" > /dev/null 2>&1 || true
    docker run -d --name "$METER" --network "$NETWORK" --env-file "$ROOT/.env" \
        -e METER_BUDGET_USD="$1" -e PYTHONPATH=/work/src -v "$ROOT":/work:ro \
        -v rag_meter_venv:/venv python:3.12.15-slim sh -c '
            [ -x /venv/bin/python ] || python -m venv /venv
            /venv/bin/pip install -q fastapi uvicorn httpx 2>&1 | grep -v -i notice || true
            exec /venv/bin/python /work/scripts/benchmark/meter_proxy.py' > /dev/null
    for _ in $(seq 1 60); do
        docker exec "$METER" /venv/bin/python -c \
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/__usage')" \
            > /dev/null 2>&1 && return 0
        sleep 2
    done
    echo "the metering proxy did not start" >&2
    docker logs "$METER" >&2 || true
    exit 1
}

meter_usage() {
    docker exec "$METER" /venv/bin/python -c \
        "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/__usage').read().decode())"
}

# meter_stop FILE: save the final usage report to FILE and remove the proxy.
meter_stop() {
    meter_usage | tee "$1"
    docker rm -f "$METER" > /dev/null 2>&1 || true
}

# bench_python_image: the image used for the benchmark's own code. It needs the packages in
# requirements.txt, kept in a named volume so they install once.
BENCH_ENV=(
    -e LLM_PROVIDER=anthropic
    -e LLM_API_KEY=via-the-metering-proxy
    -e "ANTHROPIC_BASE_URL=http://$METER:8080"
)

# Reading half a million small files through a macOS or Windows bind mount is very slow. When the
# corpus has been copied into the Docker volume below (scripts/benchmark/setup.sh --volume), the
# agent reads it from there.
SOURCES_VOLUME="rag_bench_sources"
sources_mount() {
    if docker volume inspect "$SOURCES_VOLUME" > /dev/null 2>&1; then
        echo "-v $SOURCES_VOLUME:/work/$BENCH/generated_data:ro"
    fi
}
