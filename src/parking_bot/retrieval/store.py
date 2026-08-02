"""Connect to the configured Milvus collection (Lite path or standalone URI)."""

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus

from parking_bot.config import Settings, get_settings
from parking_bot.llm.embeddings import build_embeddings


def build_vector_store(
    settings: Settings | None = None, embeddings: Embeddings | None = None
) -> Milvus:
    """Return a Milvus vector store bound to the ingested collection.

    `settings.milvus_connection_uri` picks Lite vs. standalone (see
    ADR-002), so this same call works unmodified against either.
    """
    settings = settings or get_settings()
    embeddings = embeddings or build_embeddings(settings)
    return Milvus(
        embedding_function=embeddings,
        collection_name=settings.milvus_collection,
        connection_args={"uri": settings.milvus_connection_uri},
    )
