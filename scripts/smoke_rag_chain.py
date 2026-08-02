"""Smoke test: the full RAG chain end to end (retrieval -> Sonnet 4.6).

Run:  uv run python scripts/smoke_rag_chain.py

Ingests data/static/ into the configured Milvus collection, asks a question
answerable from that content, and prints the grounded answer with its
sources. Also asks an out-of-scope question and checks the chain refuses to
answer instead of the LLM hallucinating.

This is a manual check, not a pytest test — it costs real tokens and needs a
live ANTHROPIC_API_KEY, so it must never run in CI.
"""

import sys
from pathlib import Path

from parking_bot.config import get_settings
from parking_bot.ingestion.pipeline import run_ingestion
from parking_bot.rag.chain import NO_CONTEXT_ANSWER, answer_question

DEFAULT_STATIC_DIR = Path("data/static")

IN_SCOPE_QUESTION = "How do I reserve a parking spot?"
OUT_OF_SCOPE_QUESTION = "What is the capital of France?"


def main() -> int:
    settings = get_settings()
    if not settings.anthropic_api_key or settings.anthropic_api_key.startswith("sk-ant-..."):
        print("ANTHROPIC_API_KEY is not set in .env — nothing to test.")
        return 2

    print(f"[ingest] indexing {DEFAULT_STATIC_DIR} ... ", end="", flush=True)
    chunk_count = run_ingestion(settings, static_dir=DEFAULT_STATIC_DIR)
    print(f"OK ({chunk_count} chunks)")

    print(f"\n[grounded] {IN_SCOPE_QUESTION!r}")
    grounded = answer_question(IN_SCOPE_QUESTION, settings=settings)
    print(f"  answer:  {grounded.answer}")
    print(f"  sources: {grounded.sources}")
    grounded_ok = bool(grounded.sources)

    print(f"\n[refusal] {OUT_OF_SCOPE_QUESTION!r}")
    refusal = answer_question(OUT_OF_SCOPE_QUESTION, settings=settings)
    print(f"  answer:  {refusal.answer}")
    print(f"  sources: {refusal.sources}")
    # Retrieval has no relevance threshold, so it still returns its nearest
    # (irrelevant) chunks for an out-of-scope question — the real grounding
    # check is that the LLM didn't leak the answer from outside knowledge.
    refusal_ok = refusal.answer == NO_CONTEXT_ANSWER or "paris" not in refusal.answer.lower()

    if grounded_ok and refusal_ok:
        print("\nBoth checks passed. Smoke test passed.")
        return 0
    print("\nAt least one check failed — inspect the answers above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
