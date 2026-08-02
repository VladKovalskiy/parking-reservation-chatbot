"""Prompt construction for the grounded RAG chain.

Context chunks are labeled with their doc_id so the model can (and is
instructed to) cite them, and so citations are checkable in tests without
depending on what the LLM actually generates.
"""

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are the Central Parking Garage assistant. Answer the user's question "
    "using ONLY the context sections below — never rely on outside knowledge. "
    "If the context does not contain the answer, say you don't know instead of "
    "guessing. Cite the doc_id of every context section you rely on, in the "
    "form (source: <doc_id>)."
)


def format_context(chunks: list[Document]) -> str:
    """Render retrieved chunks as labeled sections for the prompt."""
    return "\n\n".join(f"[{chunk.metadata['doc_id']}]\n{chunk.page_content}" for chunk in chunks)


def build_prompt(question: str, chunks: list[Document]) -> list[BaseMessage]:
    """Build the grounded system/human message pair for one RAG turn."""
    context = format_context(chunks)
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
