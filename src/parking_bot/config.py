"""Central configuration.

Every provider choice (LLM, embeddings, vector store, SQL) is read from the
environment so that it can be swapped without touching application code.
This is what makes the stage-1 evaluation report cheap to produce: the same
pipeline can be re-run against different embedding models by changing one
variable.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Anthropic) ---
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    # Cheaper model for high-volume internal steps (routing, extraction, guardrails)
    llm_model_fast: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.0

    # --- Embeddings ---
    # "local"  -> sentence-transformers, no API key, works offline and in CI
    # "voyage" -> Voyage AI, requires VOYAGE_API_KEY
    # "fake"   -> deterministic stub used by unit tests
    embedding_provider: Literal["local", "voyage", "fake"] = "local"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768

    # --- Vector store (Milvus) ---
    # Local file path  -> Milvus Lite, zero infrastructure (development, CI)
    # http://host:19530 -> Milvus standalone from docker/compose.yml (demo)
    milvus_uri: str = "./data/milvus_lite.db"
    milvus_collection: str = "parking_static"

    # --- Dynamic data (SQL) ---
    database_url: str = "postgresql+psycopg://parking:parking@localhost:5432/parking"

    # --- Retrieval ---
    top_k: int = Field(default=4, ge=1, le=20)
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Guardrails ---
    pii_enabled: bool = True
    pii_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor, so config is parsed once per process."""
    return Settings()
