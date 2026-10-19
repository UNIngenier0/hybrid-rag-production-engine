# Run the project on macOS, Windows and Linux

This guide gets the full stack (API, Qdrant, PostgreSQL, Redis) running on a new machine and
loads the indexed benchmark corpus into it. Every command is for a bash-style shell: the macOS
Terminal, a Linux shell, or the **Ubuntu (WSL) shell on Windows**. On Windows use PowerShell only
for the steps that say so.

## What the machine needs

| | Minimum | Better |
|---|---|---|
| RAM | 8 GB, with 6 GB given to Docker | 16 GB or more, with 11 GB or more given to Docker |
| Free disk | 45 GB | 60 GB |
| Software | Docker Desktop, Git | plus [uv](https://docs.astral.sh/uv/) to run the tests and the corpus download |

The disk figures are measured: the Qdrant volume is 15.7 GB, PostgreSQL 2.5 GB, the vector
shards 4.2 GB, the corpus 1.4 GB, and the images and caches take several GB more.

**Memory matters more than CPU.** Qdrant keeps the int8 copy of the vectors (2.35 GB) in memory
and reads the HNSW graphs, the half-precision originals and the payloads from disk through the
operating system's page cache. When the cache is too small a search takes seconds instead of
milliseconds (see [retrieval.md](retrieval.md#latency-on-the-development-laptop)). Plan on this:

| Machine RAM | Give Docker | Expectation |
|---|---|---|
| 8 GB | 6 GB | Works. Searches over cold regions of the index take seconds |
| 16 GB | 11 to 12 GB | Room for Qdrant (about 3.5 GB resident) plus several GB of page cache. Not measured yet |
| 32 GB | 20 GB or more | Should keep the whole working set cached |

## 1. Install the prerequisites

### macOS

```bash
brew install --cask docker      # then open Docker Desktop once
brew install git uv
```

### Windows 11

In PowerShell as administrator:

```powershell
wsl --install -d Ubuntu          # restart when asked, then finish the Ubuntu user setup
winget install Docker.DockerDesktop
```

Open Docker Desktop, then Settings, Resources, WSL integration, and switch on **Ubuntu**. From now
on use the Ubuntu shell, and keep the repository **inside the Ubuntu filesystem** (`~/`), not under
`/mnt/c`: the data volumes and bind mounts are many times slower across that boundary.

Install uv inside Ubuntu: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new shell.

### Linux

Install Docker Engine with the Compose plugin and Git from your distribution, and uv with the
command above.

## 2. Give Docker the memory

### macOS

Docker Desktop, Settings, Resources, Memory. Pick the value from the table above (6 GB on an 8 GB
Mac) and press Apply & restart.

### Windows

Docker Desktop uses the WSL 2 virtual machine, whose size is set in `.wslconfig`. In PowerShell:

```powershell
notepad $env:USERPROFILE\.wslconfig
```

For a 16 GB machine with 8 or more cores, put this in the file and save it:

```ini
[wsl2]
memory=12GB
processors=8
swap=4GB
```

Leave Windows at least 4 GB. Then restart WSL with `wsl --shutdown` and reopen Docker Desktop.

### Check it (every system)

```bash
docker info --format '{{.MemTotal}} bytes, {{.NCPU}} CPUs'
```

## 3. Get the code and configure it

```bash
git clone https://github.com/UNIngenier0/hybrid-rag-production-engine.git
cd hybrid-rag-production-engine
cp .env.example .env
```

Edit `.env` and choose a local password: `POSTGRES_USER`, `POSTGRES_DB` and `POSTGRES_PASSWORD`
are required. `.env` is ignored by git and must never be committed.

## 4. Start the stack

```bash
docker compose -f docker-compose.yml -f docker-compose.host-ports.yml up -d --build
docker compose ps
curl -s localhost:8000/health/ready
```

The first build downloads the images and the Python dependencies, including PyTorch; it takes a
few minutes. The override file publishes Qdrant (6333) and PostgreSQL (5432) on `127.0.0.1` only,
which the migration steps below use. Qdrant has no CPU cap here; on the Mac used for development
it was capped at two cores with `docker update --cpus 2`.

## 5. Load the data

The empty stack answers health checks but has nothing to search. Pick one path.

### A. Move the finished index from another machine (fastest)

Qdrant snapshots and PostgreSQL custom-format dumps are portable between machines and CPU
architectures (the Mac is ARM, a Windows laptop is x86). Both machines must run the same Qdrant
version, which `docker-compose.yml` pins.

**On the machine that has the index.** Needs about 35 GB of free disk during the export.

```bash
mkdir -p data/export
docker compose -f docker-compose.yml -f docker-compose.host-ports.yml up -d qdrant postgres

# PostgreSQL: about 1 GB, 2 minutes in the measured run
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
  > data/export/postgres.dump

# Qdrant: the call returns when the snapshot is written; expect several minutes
NAME=$(curl -s -X POST localhost:6333/collections/chunks_v1/snapshots \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['result']['name'])")
curl -s localhost:6333/collections/chunks_v1/snapshots/$NAME -o data/export/chunks_v1.snapshot
curl -s -X DELETE localhost:6333/collections/chunks_v1/snapshots/$NAME   # free the copy inside Qdrant
```

`data/export/` is inside the git-ignored `data/` directory. Copy the two files to the new machine
(an external drive is fine). On Windows, a USB drive appears in Ubuntu as `/mnt/d`, `/mnt/e` and
so on: `mkdir -p data/export && cp /mnt/d/export/* data/export/`.

**On the new machine,** with the stack from step 4 running:

```bash
docker compose exec -T postgres sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner --no-privileges' \
  < data/export/postgres.dump
curl -X POST 'localhost:6333/collections/chunks_v1/snapshots/upload?priority=snapshot' \
  -H 'Content-Type: multipart/form-data' -F 'snapshot=@data/export/chunks_v1.snapshot'
docker compose run --rm worker status
```

`status` should report 511,962 documents, 3,064,558 chunks and `points_without_dense` 0. The
PostgreSQL restore took 1.5 minutes and the counts matched exactly in a test. The Qdrant flow was
tested end to end on a small collection (create the snapshot, download it, upload it into an empty
Qdrant, then query it, including the sparse vector); the full 15.7 GB collection has not been moved
yet, so expect the snapshot and the upload to take a while.

### B. Rebuild the index from the vector shards

Use this when you have the shards (`data/vectors`, 4.2 GB) but not a snapshot. Copy that
directory from the first machine into `./data/vectors`, then:

```bash
uv sync                                         # the corpus download runs on the host
uv run python -m rag_engine.ingestion.download  # corpus and questions, 1.4 GB, SHA-256 checked

docker compose run --rm worker stage             # clean the corpus into PostgreSQL
docker compose run --rm worker index sparse      # chunk and BM25 into Qdrant
docker compose run --rm worker dense-graph off
docker compose run --rm worker load-vectors      # attach the GPU-computed embeddings
docker compose run --rm worker dense-graph on    # build the HNSW graph
docker compose run --rm worker status
```

On the development Mac these took 4, 42, 125 and 17 minutes, with the worker limited to two cores.
The worker is capped at two cores and 3 GB by default; on a bigger machine put `WORKER_CPUS=6` and
`WORKER_MEMORY=8g` in `.env` (see `.env.example`).

### C. From nothing

Without the shards the embeddings have to be computed again on a free GPU: follow
[kaggle-embedding.md](kaggle-embedding.md), then path B.

## 6. Try it

```bash
curl -s localhost:8000/v1/search -H 'content-type: application/json' -d '{
  "query": "What are the default size limits for file uploads and total request size?",
  "mode": "hybrid", "limit": 3, "include_context": true
}'
```

The first query takes about 25 seconds, because the API downloads and loads the embedding model
(the Compose volume `hf_cache` keeps it for next time). After that, the first minutes are the
slowest, while the page cache fills.

## 7. Run the tests (optional)

```bash
uv sync
uv run pytest                  # unit tests; no services needed
uv run ruff check . && uv run mypy
```

The `integration` tests expect the Compose stack to be running (`uv run pytest -m integration`).

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `docker info` shows less memory than you set | On Windows, `.wslconfig` was not applied: run `wsl --shutdown` and reopen Docker Desktop. On macOS, press Apply & restart |
| Qdrant restarts or disappears during a heavy load | The VM ran out of memory. Check `docker inspect <container> --format '{{.State.OOMKilled}}'`, give Docker more memory, and keep the worker at two cores |
| Searches take seconds | Cold page cache or too little memory (see the table at the top). Run a few hundred varied queries to warm it, and check `docker stats` |
| `/v1/search` answers 503 | The embedding model could not load (first run needs internet to download it) or Qdrant or PostgreSQL is unreachable. The reason is in `docker compose logs api` |
| `docker compose` complains about `POSTGRES_USER` | `.env` is missing or incomplete (step 3) |
| Errors about `\r` or "bad interpreter" on Windows | A file was edited with Windows line endings. Edit `.env` from Ubuntu (`nano .env`) and keep the repository inside the Ubuntu filesystem |
| Port 8000, 5432 or 6333 already in use | Stop the other program, or change the published port in the Compose file |

GPU note: the stack runs on the CPU, which is enough for query embedding. The PyTorch build is the
CPU one on Linux (including WSL), so an NVIDIA or AMD GPU is not used; GPU support for the
reranker can be added when it is needed.
