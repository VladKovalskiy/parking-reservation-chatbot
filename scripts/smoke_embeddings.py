"""Smoke test: embeddings + Milvus wiring (Lite or standalone).

Run:  uv run python scripts/smoke_embeddings.py
      MILVUS_URI=http://localhost:19530 uv run python scripts/smoke_embeddings.py

Loads the local sentence-transformers embedding model (downloads it on first
run), indexes 3 test documents into a Milvus collection, and checks that
similarity_search returns the expected document. Also verifies that the
multilingual-e5 query:/passage: prefixes are actually sent to the model
during encoding, not just configured and silently ignored.

The connection target comes from config (settings.milvus_connection_uri), so
this same script runs unmodified against Milvus Lite (default) or, with
MILVUS_URI set, against standalone Milvus (docker/compose.yml) — the
collection is left in place afterward so it can be inspected in Attu.

This is a manual check, not a pytest test — it downloads a large model and
touches a Milvus instance, so it must never run in CI.
"""

import sys

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from parking_bot.config import get_settings
from parking_bot.llm.embeddings import build_embeddings

DOCS = [
    Document(
        page_content="Parking spaces cost $5 per hour on weekdays and $3 per hour on weekends.",
        metadata={"id": "pricing"},
    ),
    Document(
        page_content="The parking garage is located at 123 Main Street, next to the library.",
        metadata={"id": "location"},
    ),
    Document(
        page_content="To reserve a spot, open the chat and confirm the suggested time slot.",
        metadata={"id": "booking"},
    ),
]

QUERY = "How much does it cost to park here?"
EXPECTED_ID = "pricing"


def check_prefixes(embeddings) -> bool:
    """Verify the e5 query:/passage: prefixes are actually passed to encode()."""
    seen_prompts: list[str | None] = []
    original_encode = SentenceTransformer.encode

    def spy_encode(self, *args, **kwargs):
        seen_prompts.append(kwargs.get("prompt"))
        return original_encode(self, *args, **kwargs)

    SentenceTransformer.encode = spy_encode
    try:
        embeddings.embed_documents(["a passage"])
        embeddings.embed_query("a query")
    finally:
        SentenceTransformer.encode = original_encode

    doc_prompt, query_prompt = seen_prompts
    ok = doc_prompt == "passage: " and query_prompt == "query: "
    label = "OK" if ok else "FAILED"
    print(f"[prefixes] passage={doc_prompt!r} query={query_prompt!r} ... {label}")
    return ok


def main() -> int:
    settings = get_settings()
    if settings.embedding_provider != "local":
        print(
            f"EMBEDDING_PROVIDER={settings.embedding_provider!r}, expected 'local'. "
            "Set it in .env to run this smoke test."
        )
        return 2

    print(f"[model] loading {settings.embedding_model} ... ", end="", flush=True)
    embeddings = build_embeddings(settings)
    print("OK")

    prefixes_ok = check_prefixes(embeddings)

    from langchain_milvus import Milvus

    uri = settings.milvus_connection_uri
    collection_name = f"{settings.milvus_collection}_smoke"
    print(
        f"[milvus] indexing {len(DOCS)} documents into {uri!r} "
        f"(collection={collection_name!r}) ... ",
        end="",
        flush=True,
    )
    store = Milvus.from_documents(
        DOCS,
        embeddings,
        collection_name=collection_name,
        connection_args={"uri": uri},
        drop_old=True,
    )
    print("OK")

    print(f"[search] similarity_search({QUERY!r}) ... ", end="", flush=True)
    results = store.similarity_search(QUERY, k=1)
    got_id = results[0].metadata.get("id") if results else None
    search_ok = got_id == EXPECTED_ID
    label = "OK" if search_ok else "FAILED"
    print(f"-> {got_id!r} {label}")

    if prefixes_ok and search_ok:
        print("\nAll checks passed. Smoke test passed.")
        return 0
    print("\nAt least one check failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
