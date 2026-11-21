from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_mock_response(
    content: str = "The capital of France is Paris.",
    prompt_tokens: int = 42,
    completion_tokens: int = 9,
    model: str = "gemini/gemini-2.0-flash",
) -> dict[str, Any]:
    """Build a LiteLLM-compatible ModelResponse-like dict for deterministic tests."""
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    return ModelResponse(
        id="chatcmpl-test",
        model=model,
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content=content),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@pytest.fixture()
def tmp_db_url(tmp_path: Path) -> str:
    db_file = tmp_path / "test_app.db"
    return f"sqlite:///{db_file.as_posix()}"


@pytest.fixture()
def test_settings(tmp_db_url: str) -> Settings:
    return Settings(
        llm_api_keys=["secret-test-key-alpha", "secret-test-key-beta"],
        llm_model="gemini/gemini-2.0-flash",
        database_url=tmp_db_url,
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=True,
        cache_enabled=True,
        cache_type="local",
        default_system_prompt="You are a helpful test assistant.",
    )


@pytest.fixture()
def client(test_settings: Settings) -> TestClient:
    app = create_app(test_settings)
    with TestClient(app) as test_client:
        yield test_client
    app.state.gateway.clear_cache()
