from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    keys_configured: int
    cache_enabled: bool = True


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str
    system_prompt: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        s = v.strip()
        if not (1 <= len(s) <= 128):
            raise ValueError("session_id must be between 1 and 128 characters")
        return s

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        trimmed = v.strip()
        if len(trimmed) < 1 or len(trimmed) > 4000:
            raise ValueError("message must be between 1 and 4000 characters after trim")
        return trimmed


class ChatMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str] = None
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


class MessageItem(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    metadata: Optional[Any] = None
    created_at: str


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: List[MessageItem]


class ErrorResponse(BaseModel):
    detail: Any
