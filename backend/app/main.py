from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings
from app.database import Database
from app.failure_sim import parse_simulate_failure_header
from app.llm_gateway import AllProvidersUnavailableError, LLMGateway
from app.schemas import (
    AssistantHistoryMessage,
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryMessage,
    SessionMessagesResponse,
    UserHistoryMessage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Application factory creating the FastAPI app with SQLite and LiteLLM gateway."""
    cfg = settings or Settings.from_env()
    db = Database(cfg.database_url)
    gateway = LLMGateway(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_schema()
        logger.info(
            "Backend started: keys_configured=%d cache_enabled=%s failure_simulation=%s db=%s",
            len(cfg.key_pool),
            gateway.cache_enabled,
            cfg.enable_failure_simulation,
            db.db_path,
        )
        yield
        db.close()

    app = FastAPI(
        title="Lyftr AI LLM Chatbot Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store components on app.state for clean inspection and testing
    app.state.settings = cfg
    app.state.db = db
    app.state.gateway = gateway

    allow_all = "*" in cfg.allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else cfg.allowed_origins,
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Global JSON error handlers (never return stack traces or HTML error pages) ----
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if exc.detail else "request error"
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    # ---- Exact 3 API endpoints defined by the contract ----

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        try:
            db_ok = db.ping()
        except Exception:
            db_ok = False
        if not db_ok:
            raise HTTPException(status_code=503, detail="database unavailable")

        return HealthResponse(
            status="ok",
            keys_configured=len(cfg.key_pool),
            cache_enabled=gateway.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        # Resolve effective system prompt: request override > persisted session prompt > default
        persisted_prompt = db.get_session_system_prompt(payload.session_id)
        effective_system_prompt = (
            payload.system_prompt
            or persisted_prompt
            or cfg.default_system_prompt
        )

        # Load prior multi-turn history for this session
        stored_history = db.get_messages(payload.session_id)
        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": effective_system_prompt},
            *[{"role": msg["role"], "content": msg["content"]} for msg in stored_history],
            {"role": "user", "content": payload.message},
        ]

        # Parse failover simulation header (honored only when ENABLE_FAILURE_SIMULATION=true)
        simulation_rules = (
            parse_simulate_failure_header(x_simulate_failure)
            if cfg.enable_failure_simulation
            else {}
        )

        try:
            result = await gateway.generate(
                messages=llm_messages,
                simulation_rules=simulation_rules,
            )
        except AllProvidersUnavailableError:
            raise HTTPException(status_code=503, detail="all providers unavailable")

        metadata = ChatMetadata(
            resolution_source=result.resolution_source,  # type: ignore[arg-type]
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )

        # Persist ONLY after a successful reply (failed 422/503 requests never touch history)
        db.save_exchange(
            session_id=payload.session_id,
            user_message=payload.message,
            assistant_reply=result.reply,
            assistant_metadata=metadata.model_dump(),
            system_prompt=payload.system_prompt,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=metadata,
        )

    @app.get(
        "/sessions/{session_id}/messages",
        response_model=SessionMessagesResponse,
    )
    async def get_session_messages(session_id: str) -> SessionMessagesResponse:
        rows = db.get_messages(session_id)
        if not rows:
            raise HTTPException(status_code=404, detail="session not found")

        messages: list[HistoryMessage] = []
        for row in rows:
            if row["role"] == "assistant":
                messages.append(
                    AssistantHistoryMessage(
                        role="assistant",
                        content=row["content"],
                        metadata=row.get("metadata") or {},
                    )
                )
            else:
                messages.append(
                    UserHistoryMessage(
                        role="user",
                        content=row["content"],
                    )
                )
        return SessionMessagesResponse(session_id=session_id, messages=messages)

    return app


app = create_app()
