"""Embedding backends.

The key design point: tests and CI must never need an API key or a 2 GB model
download. `FakeEmbeddings` is deterministic, so retrieval logic can be unit
tested end to end without network access.
"""

import hashlib

from langchain_core.embeddings import Embeddings

from parking_bot.config import Settings, get_settings


class FakeEmbeddings(Embeddings):
    """Deterministic hash-based embeddings for tests.

    Not semantically meaningful, but stable: the same text always maps to the
    same vector, which is all that is needed to verify wiring, filtering and
    ranking mechanics.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (digest * (self.dim // len(digest) + 1))[: self.dim]
        return [(b - 127.5) / 127.5 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()

    if settings.embedding_provider == "fake":
        return FakeEmbeddings(dim=settings.embedding_dim)

    if settings.embedding_provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        # multilingual-e5 requires "query: " / "passage: " prefixes on encoding
        # input, applied here via sentence-transformers' `prompt` encode kwarg.
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
            query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
        )

    if settings.embedding_provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(model=settings.embedding_model)

    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
