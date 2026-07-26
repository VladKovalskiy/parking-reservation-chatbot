"""Chat model factory.

Mirrors the embeddings factory: the model name comes from config, never
hardcoded, so the same code path serves both the cheap fast model (routing,
extraction, guardrails) and the main generation model.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from parking_bot.config import Settings, get_settings


def build_chat(*, fast: bool = False, settings: Settings | None = None) -> BaseChatModel:
    """Return a configured Claude chat model.

    Args:
        fast: use the cheaper model (Haiku) for high-volume internal steps
              instead of the main generation model (Sonnet).
    """
    settings = settings or get_settings()
    model = settings.llm_model_fast if fast else settings.llm_model

    return ChatAnthropic(
        model=model,
        temperature=settings.llm_temperature,
        api_key=settings.anthropic_api_key,
        timeout=30,
        max_retries=2,
    )
