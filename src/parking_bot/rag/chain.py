"""Basic RAG chain: retrieval -> grounded prompt -> Sonnet 4.6 generation.

Strict grounding is enforced two ways: the system prompt in `rag/prompt.py`
forbids answering outside the retrieved context, and — since an ungrounded
LLM call is a wasted (and hallucination-prone) one anyway — this chain never
calls the LLM at all when retrieval comes back empty; it returns a fixed
"don't know" answer instead.
"""

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from parking_bot.config import Settings, get_settings
from parking_bot.llm.chat import build_chat
from parking_bot.rag.prompt import build_prompt
from parking_bot.retrieval.retriever import retrieve

NO_CONTEXT_ANSWER = "I don't have information about that in the parking documents."


@dataclass
class RAGAnswer:
    """A generated answer plus the doc_ids of the chunks it's grounded in."""

    answer: str
    sources: list[str] = field(default_factory=list)


def answer_question(
    question: str,
    *,
    chunks: list[Document] | None = None,
    chat: BaseChatModel | None = None,
    settings: Settings | None = None,
    **retrieve_kwargs,
) -> RAGAnswer:
    """Answer `question`, grounded only in retrieved context.

    Pass `chunks` to skip retrieval (e.g. in tests, or with an
    already-retrieved batch); otherwise the configured retriever is used.
    Extra keyword arguments (`store`, `k`) are forwarded to `retrieve()`.
    """
    settings = settings or get_settings()
    if chunks is None:
        chunks = retrieve(question, settings=settings, **retrieve_kwargs)

    if not chunks:
        return RAGAnswer(answer=NO_CONTEXT_ANSWER, sources=[])

    chat = chat or build_chat(settings=settings)
    response = chat.invoke(build_prompt(question, chunks))
    content = response.content if isinstance(response.content, str) else str(response.content)

    sources = sorted({chunk.metadata["doc_id"] for chunk in chunks})
    return RAGAnswer(answer=content, sources=sources)
