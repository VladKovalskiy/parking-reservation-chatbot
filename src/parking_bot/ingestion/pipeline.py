"""Ingest data/static/ into the configured Milvus collection.

Run:  uv run python -m parking_bot.ingestion.pipeline

Idempotent: each run drops and recreates the collection (drop_old=True), so
re-running never duplicates or accumulates stale chunks — the collection
always ends up matching exactly what's currently in data/static/.
"""

import sys
from pathlib import Path

from langchain_milvus import Milvus

from parking_bot.config import Settings, get_settings
from parking_bot.ingestion.chunker import chunk_documents
from parking_bot.ingestion.loader import load_static_documents
from parking_bot.llm.embeddings import build_embeddings

DEFAULT_STATIC_DIR = Path("data/static")


def run_ingestion(settings: Settings | None = None, static_dir: Path = DEFAULT_STATIC_DIR) -> int:
    """Load, chunk, embed, and index the static documents. Returns chunk count."""
    settings = settings or get_settings()

    sections = load_static_documents(static_dir)
    chunks = chunk_documents(
        sections, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    embeddings = build_embeddings(settings)
    Milvus.from_documents(
        chunks,
        embeddings,
        collection_name=settings.milvus_collection,
        connection_args={"uri": settings.milvus_connection_uri},
        drop_old=True,
    )
    return len(chunks)


def main() -> int:
    count = run_ingestion()
    print(f"Ingested {count} chunks into Milvus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
