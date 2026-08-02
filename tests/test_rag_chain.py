from types import SimpleNamespace

from langchain_core.documents import Document

from parking_bot.rag.chain import NO_CONTEXT_ANSWER, answer_question
from parking_bot.rag.prompt import build_prompt

CHUNKS = [
    Document(
        page_content="Parking costs $5 per hour on weekdays.",
        metadata={"doc_id": "general.md#payment-methods"},
    ),
    Document(
        page_content="The garage is located at 123 Main Street.",
        metadata={"doc_id": "location.md#address"},
    ),
]


class _MockChat:
    """Records the prompt it was called with and returns a canned reply."""

    def __init__(self, content: str = "mocked answer") -> None:
        self._content = content
        self.invoked_with: list | None = None

    def invoke(self, messages: list) -> SimpleNamespace:
        self.invoked_with = messages
        return SimpleNamespace(content=self._content)


def test_build_prompt_grounds_the_question_in_labeled_context() -> None:
    messages = build_prompt("How much does parking cost?", CHUNKS)

    system_message, human_message = messages
    assert "ONLY" in system_message.content
    assert "general.md#payment-methods" in human_message.content
    assert "location.md#address" in human_message.content
    assert "How much does parking cost?" in human_message.content


def test_answer_question_sends_grounded_prompt_and_returns_sources() -> None:
    chat = _MockChat("Parking costs $5/hour (source: general.md#payment-methods)")

    result = answer_question("How much does parking cost?", chunks=CHUNKS, chat=chat)

    assert result.answer == "Parking costs $5/hour (source: general.md#payment-methods)"
    assert result.sources == ["general.md#payment-methods", "location.md#address"]
    assert chat.invoked_with is not None
    system_message, human_message = chat.invoked_with
    assert "ONLY" in system_message.content
    assert "How much does parking cost?" in human_message.content


def test_answer_question_skips_the_llm_when_retrieval_is_empty() -> None:
    chat = _MockChat()

    result = answer_question("What is the meaning of life?", chunks=[], chat=chat)

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert chat.invoked_with is None
