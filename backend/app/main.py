from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import (
    get_messages,
    get_session,
    init_db,
    save_turn,
)
from app.llm import ProviderUnavailableException, execute_chat
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionMessagesResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database schema on startup
    init_db()
    yield


# Ensure DB schema is initialized
try:
    init_db()
except Exception:
    pass

app = FastAPI(
    title="Lyftr AI LLM Chatbot",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handlers: JSON ONLY - HTML/stack traces are NEVER returned
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(ProviderUnavailableException)
async def provider_unavailable_handler(request: Request, exc: ProviderUnavailableException):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "all providers unavailable"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Exact 3 Endpoints:
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "keys_configured": len(settings.api_keys_list),
        "cache_enabled": True,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    x_simulate_failure: Optional[str] = Header(None, alias="X-Simulate-Failure"),
):
    session_id = payload.session_id
    message_content = payload.message.strip()

    # Determine system prompt:
    # 1. Check if session already exists
    existing_session = get_session(session_id)
    if payload.system_prompt is not None and payload.system_prompt.strip() != "":
        active_system_prompt = payload.system_prompt.strip()
    elif existing_session and existing_session.get("system_prompt"):
        active_system_prompt = existing_session["system_prompt"]
    else:
        active_system_prompt = settings.DEFAULT_SYSTEM_PROMPT

    # Load conversation history for this session (oldest first, excluding system prompt)
    previous_messages = get_messages(session_id)

    # Build full messages list for LiteLLM
    full_messages = [{"role": "system", "content": active_system_prompt}]
    for m in previous_messages:
        full_messages.append({"role": m["role"], "content": m["content"]})
    full_messages.append({"role": "user", "content": message_content})

    # Execute LLM call with LiteLLM cache + sequential key failover + failure simulation
    # If all fail, ProviderUnavailableException is raised -> handled by 503 handler
    # Note: Failed requests (422 or 503) MUST NOT write anything to SQLite.
    reply, metadata = execute_chat(
        model=settings.LLM_MODEL,
        messages=full_messages,
        api_keys=settings.api_keys_list,
        simulate_failure_header=x_simulate_failure,
        enable_failure_simulation=settings.ENABLE_FAILURE_SIMULATION,
    )

    # Only upon successful inference, write to SQLite
    save_turn(
        session_id=session_id,
        system_prompt=active_system_prompt,
        user_content=message_content,
        assistant_content=reply,
        metadata=metadata,
    )

    return {
        "session_id": session_id,
        "reply": reply,
        "metadata": metadata,
    }


@app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="session not found",
        )

    messages = get_messages(session_id)
    return {
        "session_id": session_id,
        "messages": messages,
    }
