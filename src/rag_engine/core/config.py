"""Application settings, read from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    qdrant_url: str = "http://localhost:6333"
    # gRPC is much faster than REST for loading millions of vectors.
    qdrant_prefer_grpc: bool = False
    qdrant_grpc_port: int = 6334
    redis_url: str = "redis://localhost:6379/0"
    # Required and secret: it carries credentials, so there is deliberately no default.
    database_url: SecretStr

    health_check_timeout_seconds: float = Field(default=2.0, gt=0)

    # Index. The collection name carries the index configuration (model, chunking), because
    # changing either one means re-embedding everything into a new collection.
    qdrant_collection: str = "chunks_v1"
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m-v1.5"
    embedding_revision: str | None = "e58a8f756156a1293d763f17e3aae643474e9b8a"
    embedding_dimension: int = Field(default=768, gt=0)
    embedding_device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    embedding_batch_size: int = Field(default=128, gt=0)
    # Mean BM25 document length in terms, after stop-word removal and stemming. Measured on a
    # stratified sample of the benchmark chunks (docs/reports/corpus-profile.json).
    bm25_avg_len: float = Field(default=101.0, gt=0)

    # Chunking, in characters (about 3.7 characters per token on this corpus).
    chunk_target_chars: int = Field(default=1000, gt=0)
    chunk_max_chars: int = Field(default=1500, gt=0)
    chunk_min_chars: int = Field(default=150, gt=0)
    chunk_overlap_chars: int = Field(default=120, ge=0)
    chunk_parent_chars: int = Field(default=3200, gt=0)

    # Retrieval. Each retriever returns its `*_top_k` best chunks and the fused list keeps the
    # first `fusion_top_k`. Candidate depth is measured in docs/ingestion-report.md: more candidates
    # raise the recall a reranker can reach, at the price of reranking work.
    dense_top_k: int = Field(default=50, gt=0)
    sparse_top_k: int = Field(default=50, gt=0)
    fusion_top_k: int = Field(default=30, gt=0)
    qdrant_timeout_seconds: int = Field(default=30, gt=0)
    # HNSW search effort for dense queries. 64 instead of Qdrant's default 128 answered cold
    # queries several times faster for 0.3 points of recall (docs/ingestion-report.md).
    qdrant_hnsw_ef: int | None = Field(default=64, gt=0)
    # Reranking. The cross-encoder rescores the first `rerank_top_k` fused chunks. No model is
    # loaded unless one is named: BAAI/bge-reranker-v2-m3 was the most accurate in the ablation but
    # needs a GPU, BAAI/bge-reranker-base is the lighter choice (see docs/evaluation.md).
    reranker_model: str | None = None
    rerank_top_k: int = Field(default=30, gt=0)
    # Reciprocal Rank Fusion: score = weight / (rrf_k + rank). BM25 is the stronger retriever on
    # this corpus, and a weight of 1.5 gained recall on both halves of the evaluation questions
    # (eval/reports/retrieval-ablation.md).
    rrf_k: int = Field(default=60, gt=0)
    rrf_dense_weight: float = Field(default=1.0, gt=0)
    rrf_sparse_weight: float = Field(default=1.5, gt=0)
    # The BM25 weight when the fused list feeds the reranker. The reranker study and the answer
    # evaluation were run with equal weights, and a heavier BM25 showed no gain once the
    # cross-encoder reorders the pool, so the pipeline that ships is the one that was measured.
    rerank_sparse_weight: float = Field(default=1.0, gt=0)

    # Answer generation. The key is a secret and stays in the git-ignored .env. Haiku is the default
    # because it is cheap and good enough to compare retrieval; a stronger model is a setting away.
    anthropic_api_key: SecretStr | None = None
    # Only to reach the API through a metering proxy, as the benchmark scripts do.
    anthropic_base_url: str | None = None
    llm_model: str = "claude-haiku-5-5"
    # Newer models may reason before they write, and that reasoning counts against the limit.
    llm_max_tokens: int = Field(default=4000, gt=0)
    # POST /v1/query. The answer is written from this many passages (the reranked best ones).
    query_chunks: int = Field(default=5, ge=1, le=10)
    # A judge model checks that each cited statement is in the passage it cites. It defaults to the
    # answering model; a stronger one makes the check stricter and costs more.
    verify_citations: bool = True
    verification_model: str | None = None
    # Decline to answer when the best reranked passage scores below this (cross-encoder score of
    # the configured reranker, so it only means something with `hybrid+rerank`). None never
    # declines on score.
    abstain_threshold: float | None = None

    # Bulk ingestion.
    dataset_path: Path = Path("data/raw/documents/test.parquet")
    vectors_dir: Path = Path("data/vectors")
    ingest_batch_size: int = Field(default=200, gt=0)

    @field_validator("anthropic_api_key", "verification_model", "abstain_threshold", mode="before")
    @classmethod
    def _empty_means_unset(cls, value: object) -> object:
        """Docker Compose passes an unset variable as an empty string: that means "not set"."""
        return None if isinstance(value, str) and not value.strip() else value


@lru_cache
def get_settings() -> Settings:
    # Required fields such as database_url are read from the environment, not passed here.
    return Settings()  # type: ignore[call-arg]
