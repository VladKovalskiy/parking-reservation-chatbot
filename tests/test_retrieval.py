from langchain_core.documents import Document
from langchain_milvus import Milvus

from parking_bot.config import Settings
from parking_bot.llm.embeddings import build_embeddings
from parking_bot.retrieval.retriever import retrieve

DOCS = [
    Document(
        page_content="Parking spaces cost $5 per hour on weekdays.", metadata={"id": "pricing"}
    ),
    Document(page_content="The garage is located at 123 Main Street.", metadata={"id": "location"}),
    Document(
        page_content="Reservations require explicit confirmation.", metadata={"id": "booking"}
    ),
]


def _seeded_store(settings: Settings) -> Milvus:
    embeddings = build_embeddings(settings)
    return Milvus.from_documents(
        DOCS,
        embeddings,
        collection_name=settings.milvus_collection,
        connection_args={"uri": settings.milvus_connection_uri},
        drop_old=True,
    )


def test_retrieve_ranks_the_exact_matching_chunk_first(settings: Settings) -> None:
    store = _seeded_store(settings)

    results = retrieve(DOCS[1].page_content, store=store, k=3)

    assert results[0].metadata["id"] == "location"


def test_retrieve_applies_top_k(settings: Settings) -> None:
    store = _seeded_store(settings)

    results = retrieve("parking", store=store, k=2)

    assert len(results) == 2
