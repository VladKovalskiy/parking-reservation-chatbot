"""Request/response models for the chat interface."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(
        ..., min_length=1, description="Chat/session identity (users.external_id)"
    )


class ChatResponse(BaseModel):
    answer: str
    source: str = Field(..., description="'sql' or 'rag' — which path produced the answer")
    sources: list[str] = Field(default_factory=list)
