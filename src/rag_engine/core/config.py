"""Application settings, read from environment variables and an optional .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"
    # Required and secret: it carries credentials, so there is deliberately no default.
    database_url: SecretStr

    health_check_timeout_seconds: float = Field(default=2.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    # Required fields such as database_url are read from the environment, not passed here.
    return Settings()  # type: ignore[call-arg]
