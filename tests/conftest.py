from collections.abc import Callable

import httpx
import pytest

from rag_engine.api.main import create_app
from rag_engine.core.config import Settings
from rag_engine.core.ports import HealthProbe


@pytest.fixture
def settings() -> Settings:
    # `_env_file=None` keeps a developer's local .env out of the tests.
    return Settings(_env_file=None, database_url="postgresql://rag:s3cret@db:5432/rag")


@pytest.fixture
def make_client(settings: Settings) -> Callable[..., httpx.AsyncClient]:
    """Build an in-process HTTP client for an app wired with the given probes."""

    def _make(*probes: HealthProbe, timeout_seconds: float | None = None) -> httpx.AsyncClient:
        app_settings = settings
        if timeout_seconds is not None:
            app_settings = settings.model_copy(
                update={"health_check_timeout_seconds": timeout_seconds}
            )
        app = create_app(app_settings, probes)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    return _make
