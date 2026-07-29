from collections.abc import Iterator

import pytest

from parking_bot.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force every test into an offline, key-free configuration."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("MILVUS_LITE_PATH", ":memory:")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return get_settings()
