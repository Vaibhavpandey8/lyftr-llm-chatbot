from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok"]
    keys_configured: int
    cache_enabled: bool


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., description="Client-generated session identifier (1-128 chars)")
    message: str = Field(..., description="User message (1-4000 chars after trimming whitespace)")
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt override for this session",
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("session_id must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("session_id must not be empty")
        if len(value) > 128:
            raise ValueError("session_id must be at most 128 characters")
        return value

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("message must be a string")
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message must not be empty")
        if len(trimmed) > 4000:
            raise ValueError("message must be at most 4000 characters after trimming whitespace")
        return trimmed

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("system_prompt must be a string")
        trimmed = value.strip()
        if not trimmed:
            return None
        if len(trimmed) > 4000:
            raise ValueError("system_prompt must be at most 4000 characters")
        return trimmed


class ChatMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


class UserHistoryMessage(BaseModel):
    role: Literal["user"]
    content: str


class AssistantHistoryMessage(BaseModel):
    role: Literal["assistant"]
    content: str
    metadata: dict[str, Any]


HistoryMessage = UserHistoryMessage | AssistantHistoryMessage


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
