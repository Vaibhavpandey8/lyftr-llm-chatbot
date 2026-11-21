from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import litellm
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import make_mock_response


def test_health_endpoint(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["keys_configured"] == 2
    assert body["cache_enabled"] is True


def test_validation_errors_422_and_no_history_saved(client: TestClient) -> None:
    # Missing message
    r1 = client.post("/chat", json={"session_id": "s-val"})
    assert r1.status_code == 422
    assert "detail" in r1.json()

    # Empty whitespace message
    r2 = client.post("/chat", json={"session_id": "s-val", "message": "    "})
    assert r2.status_code == 422
    assert "detail" in r2.json()

    # Message > 4000 chars
    r3 = client.post("/chat", json={"session_id": "s-val", "message": "a" * 4001})
    assert r3.status_code == 422
    assert "detail" in r3.json()

    # Empty session_id
    r4 = client.post("/chat", json={"session_id": "", "message": "hello"})
    assert r4.status_code == 422
    assert "detail" in r4.json()

    # session_id > 128 chars
    r5 = client.post("/chat", json={"session_id": "s" * 129, "message": "hello"})
    assert r5.status_code == 422
    assert "detail" in r5.json()

    # Malformed JSON
    r6 = client.post(
        "/chat",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r6.status_code == 422
    assert "detail" in r6.json()

    # Ensure nothing was saved for s-val
    hist = client.get("/sessions/s-val/messages")
    assert hist.status_code == 404
    assert hist.json() == {"detail": "session not found"}


def test_unknown_session_returns_404(client: TestClient) -> None:
    res = client.get("/sessions/does-not-exist/messages")
    assert res.status_code == 404
    assert res.json() == {"detail": "session not found"}


def test_chat_live_inference_and_cross_session_cache_hit(
    client: TestClient,
    monkeypatch: Any,
) -> None:
    async def slow_completion(**kwargs: Any):
        await asyncio.sleep(0.05)  # 50ms simulated network call
        return make_mock_response(
            content="The capital of France is Paris.",
            prompt_tokens=42,
            completion_tokens=9,
            model=kwargs["model"],
        )

    mock_acompletion = AsyncMock(side_effect=slow_completion)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    # First call in session 1 -> LIVE_INFERENCE
    r1 = client.post(
        "/chat",
        json={"session_id": "s-cache-1", "message": "What is the capital of France?"},
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["session_id"] == "s-cache-1"
    assert d1["reply"] == "The capital of France is Paris."
    m1 = d1["metadata"]
    assert m1["resolution_source"] == "LIVE_INFERENCE"
    assert m1["model"] == "gemini/gemini-2.0-flash"
    assert isinstance(m1["latency_ms"], int) and m1["latency_ms"] >= 30
    assert m1["input_tokens"] == 42
    assert m1["output_tokens"] == 9
    assert m1["key_id"] == "key_1"
    assert m1["fallback_attempts"] == 0
    assert mock_acompletion.call_count == 1
    # Never leak actual API key
    assert "secret-test-key" not in r1.text

    # Same first message in a NEW session -> LLM_CACHE (identical reply, key_id=null, faster)
    r2 = client.post(
        "/chat",
        json={"session_id": "s-cache-2", "message": "What is the capital of France?"},
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["session_id"] == "s-cache-2"
    assert d2["reply"] == d1["reply"]
    m2 = d2["metadata"]
    assert m2["resolution_source"] == "LLM_CACHE"
    assert m2["model"] == "gemini/gemini-2.0-flash"
    assert m2["input_tokens"] == 42
    assert m2["output_tokens"] == 9
    assert m2["key_id"] is None
    assert m2["fallback_attempts"] == 0
    assert m2["latency_ms"] < m1["latency_ms"]
    assert mock_acompletion.call_count == 1  # No extra upstream LLM call

    # Different message in a new session must NEVER come from cache
    r3 = client.post(
        "/chat",
        json={"session_id": "s-cache-3", "message": "What is the capital of Japan?"},
    )
    assert r3.status_code == 200
    assert r3.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert mock_acompletion.call_count == 2


def test_multi_turn_state_and_history_endpoint(
    client: TestClient,
    monkeypatch: Any,
) -> None:
    captured_messages: list[list[dict[str, str]]] = []

    async def fake_completion(**kwargs: Any):
        msgs = kwargs["messages"]
        captured_messages.append(msgs)
        turn = len(captured_messages)
        return make_mock_response(
            content=f"Reply {turn}",
            prompt_tokens=20 * turn,
            completion_tokens=5 + turn,
            model=kwargs["model"],
        )

    monkeypatch.setattr(litellm, "acompletion", AsyncMock(side_effect=fake_completion))

    r1 = client.post(
        "/chat",
        json={
            "session_id": "s-multi",
            "message": "My name is Ada.",
            "system_prompt": "Always be brief.",
        },
    )
    assert r1.status_code == 200
    assert r1.json()["metadata"]["input_tokens"] == 20
    assert r1.json()["metadata"]["output_tokens"] == 6

    # Turn 2 omits system_prompt -> must remember system prompt AND turn 1 history
    r2 = client.post(
        "/chat",
        json={"session_id": "s-multi", "message": "What is my name?"},
    )
    assert r2.status_code == 200
    assert r2.json()["metadata"]["input_tokens"] == 40
    assert r2.json()["metadata"]["output_tokens"] == 7

    assert captured_messages[0] == [
        {"role": "system", "content": "Always be brief."},
        {"role": "user", "content": "My name is Ada."},
    ]
    assert captured_messages[1] == [
        {"role": "system", "content": "Always be brief."},
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "What is my name?"},
    ]

    # Verify GET /sessions/{session_id}/messages returns oldest first, without system prompt
    hist = client.get("/sessions/s-multi/messages")
    assert hist.status_code == 200
    hdata = hist.json()
    assert hdata["session_id"] == "s-multi"
    assert len(hdata["messages"]) == 4
    assert all(m["role"] != "system" for m in hdata["messages"])
    assert hdata["messages"][0] == {"role": "user", "content": "My name is Ada."}
    assert hdata["messages"][1]["role"] == "assistant"
    assert hdata["messages"][1]["content"] == "Reply 1"
    assert hdata["messages"][1]["metadata"]["input_tokens"] == 20
    assert hdata["messages"][1]["metadata"]["output_tokens"] == 6
    assert hdata["messages"][2] == {"role": "user", "content": "What is my name?"}
    assert hdata["messages"][3]["content"] == "Reply 2"


def test_simulated_failover_headers(
    client: TestClient,
    monkeypatch: Any,
) -> None:
    seen_api_keys: list[str] = []

    async def fake_completion(**kwargs: Any):
        seen_api_keys.append(kwargs["api_key"])
        return make_mock_response(
            content="Recovered on backup key.",
            prompt_tokens=30,
            completion_tokens=6,
            model=kwargs["model"],
        )

    monkeypatch.setattr(litellm, "acompletion", AsyncMock(side_effect=fake_completion))

    # key_1=429 -> fails over to key_2
    r1 = client.post(
        "/chat",
        json={"session_id": "s-fail-1", "message": "Test 429 failover"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )
    assert r1.status_code == 200
    m1 = r1.json()["metadata"]
    assert m1["resolution_source"] == "LIVE_INFERENCE"
    assert m1["key_id"] == "key_2"
    assert m1["fallback_attempts"] == 1
    assert seen_api_keys[-1] == "secret-test-key-beta"

    # key_1=500 -> fails over to key_2 (even for same message!)
    r2 = client.post(
        "/chat",
        json={"session_id": "s-fail-2", "message": "Test 429 failover"},
        headers={"X-Simulate-Failure": "key_1=500"},
    )
    assert r2.status_code == 200
    m2 = r2.json()["metadata"]
    assert m2["key_id"] == "key_2"
    assert m2["fallback_attempts"] == 1

    # *=429 -> 503 all providers unavailable, nothing added to history
    r3 = client.post(
        "/chat",
        json={"session_id": "s-fail-all", "message": "Will fail completely"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert r3.status_code == 503
    assert r3.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/s-fail-all/messages").status_code == 404

    # Subsequent normal request without header STILL uses key_1 (no poisoned cooldown)
    r4 = client.post(
        "/chat",
        json={"session_id": "s-normal", "message": "Fresh normal prompt"},
    )
    assert r4.status_code == 200
    assert r4.json()["metadata"]["key_id"] == "key_1"
    assert r4.json()["metadata"]["fallback_attempts"] == 0
    assert seen_api_keys[-1] == "secret-test-key-alpha"


def test_real_provider_429_and_5xx_failover(
    client: TestClient,
    monkeypatch: Any,
) -> None:
    call_num = 0

    async def flaky_completion(**kwargs: Any):
        nonlocal call_num
        call_num += 1
        if kwargs["api_key"] == "secret-test-key-alpha":
            raise litellm.RateLimitError(
                message="429 Resource Exhausted",
                llm_provider="gemini",
                model=kwargs["model"],
            )
        return make_mock_response(
            content="Served by key_2 after real 429.",
            prompt_tokens=25,
            completion_tokens=8,
            model=kwargs["model"],
        )

    monkeypatch.setattr(litellm, "acompletion", AsyncMock(side_effect=flaky_completion))

    r = client.post(
        "/chat",
        json={"session_id": "s-real-failover", "message": "Trigger real failover"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["reply"] == "Served by key_2 after real 429."
    assert data["metadata"]["key_id"] == "key_2"
    assert data["metadata"]["fallback_attempts"] == 1
    assert call_num == 2


def test_simulation_header_ignored_when_disabled(tmp_db_url: str, monkeypatch: Any) -> None:
    cfg = Settings(
        llm_api_keys=["k1", "k2"],
        llm_model="gemini/gemini-2.0-flash",
        database_url=tmp_db_url,
        enable_failure_simulation=False,
        cache_enabled=True,
    )
    app = create_app(cfg)

    monkeypatch.setattr(
        litellm,
        "acompletion",
        AsyncMock(return_value=make_mock_response(content="Normal reply from key_1")),
    )

    with TestClient(app) as tc:
        res = tc.post(
            "/chat",
            json={"session_id": "s-sim-off", "message": "Hello"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert res.status_code == 200
        assert res.json()["metadata"]["key_id"] == "key_1"
        assert res.json()["metadata"]["fallback_attempts"] == 0
    app.state.gateway.clear_cache()
