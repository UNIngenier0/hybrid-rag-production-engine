# syntax=docker/dockerfile:1

# --- Build stage: resolve and install locked dependencies into a virtualenv ---
FROM python:3.12.15-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so this layer is cached until the lockfile changes. The `embeddings` group
# (torch, CPU build) is there because the API embeds each query with the same model as the index.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group embeddings --no-install-project

# Then the project itself.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group embeddings --no-editable

# --- Runtime stage: only the virtualenv, running as a non-root user. The API and the bulk-ingestion
# worker share this image; the worker only overrides the entrypoint in docker-compose.yml. ---
FROM python:3.12.15-slim AS runtime

RUN useradd --system --uid 10001 --no-create-home app

COPY --from=builder /app/.venv /app/.venv

# Model downloads (the embedding model, the BM25 term lists) are cached here; the API and the
# worker mount a volume on it.
RUN mkdir -p /cache/huggingface && chown -R app /cache

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface

USER app
EXPOSE 8000

CMD ["uvicorn", "rag_engine.api.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
