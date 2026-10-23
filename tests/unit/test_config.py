import pytest
from pydantic import ValidationError

from rag_engine.core.config import Settings


def test_defaults(settings: Settings) -> None:
    assert settings.log_level == "INFO"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.health_check_timeout_seconds == 2.0


def test_values_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    loaded = Settings(_env_file=None)

    assert loaded.database_url.get_secret_value() == "postgresql://u:p@host/db"
    assert loaded.qdrant_url == "http://qdrant:6333"
    assert loaded.log_level == "DEBUG"


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None)


def test_credentials_do_not_appear_in_repr(settings: Settings) -> None:
    assert "s3cret" not in repr(settings)
    assert "s3cret" not in str(settings)


def test_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None, database_url="postgresql://u:p@h/d", health_check_timeout_seconds=0
        )


def test_retrieval_defaults_match_the_documented_design(settings: Settings) -> None:
    assert (settings.dense_top_k, settings.sparse_top_k, settings.fusion_top_k) == (50, 50, 30)
    assert settings.rrf_k == 60
    assert (settings.rrf_dense_weight, settings.rrf_sparse_weight) == (1.0, 1.5)  # BM25 counts more
    assert settings.rerank_sparse_weight == 1.0  # the weight the reranker study measured
    assert settings.reranker_model is None  # no model is loaded unless one is named
    assert settings.rerank_top_k == 30


@pytest.mark.parametrize("field", ["dense_top_k", "sparse_top_k", "fusion_top_k", "rrf_k"])
def test_retrieval_depths_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="postgresql://u:p@h/d", **{field: 0})


def test_dense_search_effort_defaults_to_the_measured_value(settings: Settings) -> None:
    assert settings.qdrant_hnsw_ef == 64


def test_the_abstention_threshold_is_off_unless_it_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    monkeypatch.delenv("ABSTAIN_THRESHOLD", raising=False)
    assert Settings(_env_file=None).abstain_threshold is None

    monkeypatch.setenv("ABSTAIN_THRESHOLD", "")  # what Docker Compose passes for an unset variable
    assert Settings(_env_file=None).abstain_threshold is None

    monkeypatch.setenv("ABSTAIN_THRESHOLD", "-1.5")
    assert Settings(_env_file=None).abstain_threshold == -1.5
