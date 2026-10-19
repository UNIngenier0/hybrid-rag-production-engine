"""Application settings, read from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
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

    # Bulk ingestion.
    dataset_path: Path = Path("data/raw/documents/test.parquet")
    vectors_dir: Path = Path("data/vectors")
    ingest_batch_size: int = Field(default=200, gt=0)


@lru_cache
def get_settings() -> Settings:
    # Required fields such as database_url are read from the environment, not passed here.
    return Settings()  # type: ignore[call-arg]
