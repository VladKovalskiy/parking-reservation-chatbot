"""Top-k similarity retrieval over the Milvus vector store.

The query text is embedded through the configured embeddings model, which
applies the e5 "query: " prefix internally for the local provider (see
llm/embeddings.py's query_encode_kwargs) — retrieval code never touches the
prefix directly, it just calls similarity_search with plain query text.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from parking_bot.config import Settings, get_settings
from parking_bot.retrieval.store import build_vector_store


def retrieve(
    query: str,
    *,
    store: VectorStore | None = None,
    k: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Return the top-k chunks most relevant to `query`, ranked by similarity."""
    settings = settings or get_settings()
    store = store or build_vector_store(settings)
    k = k if k is not None else settings.top_k
    return store.similarity_search(query, k=k)
