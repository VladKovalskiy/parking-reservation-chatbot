from parking_bot.config import Settings
from parking_bot.llm.embeddings import FakeEmbeddings, build_embeddings


def test_settings_read_environment_overrides(settings: Settings) -> None:
    """Config must come from the environment, never be hardcoded."""
    assert settings.embedding_provider == "fake"
    assert settings.milvus_uri == ":memory:"


def test_fake_embeddings_are_deterministic_and_correctly_sized() -> None:
    """Same text -> same vector, so retrieval tests are reproducible."""
    emb = FakeEmbeddings(dim=768)
    first = emb.embed_query("How much does parking cost per day?")
    second = emb.embed_query("How much does parking cost per day?")

    assert first == second
    assert len(first) == 768
    assert all(-1.0 <= value <= 1.0 for value in first)


def test_factory_returns_fake_backend_when_configured(settings: Settings) -> None:
    assert isinstance(build_embeddings(settings), FakeEmbeddings)
