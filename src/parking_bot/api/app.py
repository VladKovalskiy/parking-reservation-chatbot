"""Chat interface: routing -> RAG/SQL (with PII guardrails around the RAG
leg) -> response.

Run:  uv run uvicorn parking_bot.api.app:app --reload

Guardrails live inside `rag.router.answer_dynamic_question()`, not here —
see that function's docstring for why masking has to happen *after*
classification (masking the raw question first breaks routing itself: e.g.
"Working hours?" gets NER-tagged as a single DATE_TIME span, which loses
the word "hours" the classifier needs) and only around the RAG branch, not
the SQL one (a SQL answer is our own deterministic template over DB
columns, never free text). This endpoint just wires the HTTP request to
that function and back.
"""

from fastapi import Depends, FastAPI
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from parking_bot.api.dependencies import get_chat_model, get_db_session, get_rag_chunks
from parking_bot.api.schemas import ChatRequest, ChatResponse
from parking_bot.config import Settings, get_settings
from parking_bot.rag.router import answer_dynamic_question

app = FastAPI(title="Parking Reservation Chatbot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    session: Session = Depends(get_db_session),
    chat_model: BaseChatModel | None = Depends(get_chat_model),
    rag_chunks: list[Document] | None = Depends(get_rag_chunks),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    rag_kwargs = {} if rag_chunks is None else {"chunks": rag_chunks}
    result = answer_dynamic_question(
        request.message, session, chat=chat_model, settings=settings, **rag_kwargs
    )

    return ChatResponse(answer=result.answer, source=result.destination, sources=result.sources)
