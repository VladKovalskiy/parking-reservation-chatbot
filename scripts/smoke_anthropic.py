"""Smoke test: verify Anthropic API access for both configured models.

Run:  uv run python scripts/smoke_anthropic.py

Reads ANTHROPIC_API_KEY and model names from .env via the project config.
This is a manual check, not a pytest test — it costs real tokens and needs
a live key, so it must never run in CI.
"""

import sys

from parking_bot.config import get_settings
from parking_bot.llm.chat import build_chat


def check(label: str, *, fast: bool) -> bool:
    settings = get_settings()
    model = settings.llm_model_fast if fast else settings.llm_model
    print(f"[{label}] model={model} ... ", end="", flush=True)
    try:
        llm = build_chat(fast=fast)
        reply = llm.invoke("Reply with exactly one word: pong")
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        print(f"OK -> {text.strip()!r}")
        return True
    except Exception as exc:  # noqa: BLE001 - smoke test wants the raw reason
        print(f"FAILED -> {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    settings = get_settings()
    if not settings.anthropic_api_key or settings.anthropic_api_key.startswith("sk-ant-..."):
        print("ANTHROPIC_API_KEY is not set in .env — nothing to test.")
        return 2

    ok_main = check("main", fast=False)
    ok_fast = check("fast", fast=True)

    if ok_main and ok_fast:
        print("\nBoth models reachable. Smoke test passed.")
        return 0
    print("\nAt least one model failed. Check the key, credits, and model names.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
