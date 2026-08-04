"""FastAPI dependencies for the chat interface.

Each one mirrors a parameter the underlying chain already accepts as an
override for testing (`rag/chain.py`'s `chat`/`chunks`, `db/base.py`'s
session factory) — overriding them via `app.dependency_overrides` in tests
keeps the interface layer testable without a live Anthropic key, Milvus, or
Postgres, same "everything swappable" principle as the rest of the project
(ADR-003).
"""

from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from parking_bot.config import get_settings
from parking_bot.db.base import build_engine, build_session_factory


def get_db_session() -> Iterator[Session]:
    settings = get_settings()
    session_factory = build_session_factory(build_engine(settings))
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def get_chat_model() -> BaseChatModel | None:
    """None means "use the configured default" (`llm.chat.build_chat()`)."""
    return None


def get_rag_chunks() -> list[Document] | None:
    """None means "run real retrieval"; override with a fixed list in tests."""
    return None
