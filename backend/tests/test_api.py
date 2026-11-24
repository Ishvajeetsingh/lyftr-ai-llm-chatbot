import uuid
import pytest
from app.config import settings
from app.database import get_messages, get_session
from app.llm import _LOCAL_COMPLETION_CACHE


def test_health_endpoint(client):
    """Test GET /health returns exact structure and 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["keys_configured"] == 3
    assert data["cache_enabled"] is True


def test_validation_422_empty_or_too_long(client):
    """Test 422 validation on invalid session_id or message; verify nothing in SQLite."""
    session_id = f"test-val-{uuid.uuid4().hex[:8]}"

    # Empty trimmed message
    res = client.post("/chat", json={"session_id": session_id, "message": "   "})
    assert res.status_code == 422
    assert "detail" in res.json()

    # Message exceeding 4000 chars
    long_msg = "x" * 4001
    res = client.post("/chat", json={"session_id": session_id, "message": long_msg})
    assert res.status_code == 422

    # Session ID empty
    res = client.post("/chat", json={"session_id": "", "message": "Valid message"})
    assert res.status_code == 422

    # Session ID > 128 chars
    long_session = "a" * 129
    res = client.post("/chat", json={"session_id": long_session, "message": "Valid message"})
    assert res.status_code == 422

    # Verify nothing was written to SQLite
    assert get_session(session_id) is None
    assert get_messages(session_id) == []

    # GET /sessions/{session_id}/messages must return 404
    res_get = client.get(f"/sessions/{session_id}/messages")
    assert res_get.status_code == 404
    assert res_get.json() == {"detail": "session not found"}


def test_session_not_found_404(client):
    """Test GET /sessions/{session_id}/messages returns 404 for nonexistent session."""
    res_get = client.get("/sessions/nonexistent-session-xyz/messages")
    assert res_get.status_code == 404
    assert res_get.json() == {"detail": "session not found"}


def test_multi_turn_sqlite_history(client):
    """Test multi-turn chat persistence in SQLite with session_prompt excluded from messages."""
    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    system_prompt = "You are a specialized code reviewer."

    # Turn 1
    res1 = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "First user message",
            "system_prompt": system_prompt,
        },
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["session_id"] == session_id
    assert "reply" in data1
    assert data1["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    # Turn 2
    res2 = client.post(
        "/chat",
        json={"session_id": session_id, "message": "Second user follow-up"},
    )
    assert res2.status_code == 200
    data2 = res2.json()

    # Retrieve messages via GET /sessions/{session_id}/messages
    history_res = client.get(f"/sessions/{session_id}/messages")
    assert history_res.status_code == 200
    history = history_res.json()
    assert history["session_id"] == session_id
    msgs = history["messages"]

    # Must contain 4 messages: user, assistant, user, assistant (system prompt excluded)
    assert len(msgs) == 4
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "First user message"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == data1["reply"]
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == "Second user follow-up"
    assert msgs[3]["role"] == "assistant"
    assert msgs[3]["content"] == data2["reply"]

    # Verify sessions table persisted system prompt
    sess_row = get_session(session_id)
    assert sess_row is not None
    assert sess_row["system_prompt"] == system_prompt


def test_cache_hit_vs_miss(client):
    """
    Test LiteLLM cache:
    Identical first prompt + system prompt in a new session returns LLM_CACHE,
    identical reply, stored tokens, key_id: null, fallback_attempts: 0.
    """
    _LOCAL_COMPLETION_CACHE.clear()

    sess_1 = f"c1-{uuid.uuid4().hex[:8]}"
    sess_2 = f"c2-{uuid.uuid4().hex[:8]}"
    msg = "Explain quantum computing in one sentence"
    sys_prompt = "You are a physics professor."

    # First request: Cache Miss -> LIVE_INFERENCE
    res1 = client.post(
        "/chat",
        json={"session_id": sess_1, "message": msg, "system_prompt": sys_prompt},
    )
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert d1["metadata"]["key_id"] == "key_1"
    assert d1["metadata"]["fallback_attempts"] == 0
    first_reply = d1["reply"]
    first_in_tokens = d1["metadata"]["input_tokens"]
    first_out_tokens = d1["metadata"]["output_tokens"]

    # Second request in a brand new session with IDENTICAL prompt + system prompt
    res2 = client.post(
        "/chat",
        json={"session_id": sess_2, "message": msg, "system_prompt": sys_prompt},
    )
    assert res2.status_code == 200
    d2 = res2.json()

    # Must return LLM_CACHE, identical reply, stored tokens, key_id: null, fallback_attempts: 0
    assert d2["metadata"]["resolution_source"] == "LLM_CACHE"
    assert d2["metadata"]["key_id"] is None
    assert d2["metadata"]["fallback_attempts"] == 0
    assert d2["reply"] == first_reply
    assert d2["metadata"]["input_tokens"] == first_in_tokens
    assert d2["metadata"]["output_tokens"] == first_out_tokens
    assert isinstance(d2["metadata"]["latency_ms"], int)


def test_per_request_tokens_not_cumulative(client):
    """Verify input_tokens and output_tokens are for THIS request only, never cumulative."""
    session_id = f"tok-{uuid.uuid4().hex[:8]}"

    # Turn 1
    res1 = client.post(
        "/chat",
        json={"session_id": session_id, "message": "Short query"},
    )
    assert res1.status_code == 200
    meta1 = res1.json()["metadata"]
    t1_in = meta1["input_tokens"]
    t1_out = meta1["output_tokens"]

    # Turn 2 with different length
    res2 = client.post(
        "/chat",
        json={"session_id": session_id, "message": "A slightly longer query for turn two"},
    )
    assert res2.status_code == 200
    meta2 = res2.json()["metadata"]
    t2_in = meta2["input_tokens"]
    t2_out = meta2["output_tokens"]

    # Tokens must not be cumulative (output_tokens is per-request)
    assert t2_out == 8
    assert t2_out != (t1_out + 8)


def test_simulate_failure_key_1_429(client):
    """Test X-Simulate-Failure: key_1=429 triggers failover to key_2."""
    session_id = f"fail1-{uuid.uuid4().hex[:8]}"
    res = client.post(
        "/chat",
        json={"session_id": session_id, "message": "Test failover 429"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["metadata"]["key_id"] == "key_2"
    assert data["metadata"]["fallback_attempts"] == 1


def test_simulate_failure_key_1_500(client):
    """Test X-Simulate-Failure: key_1=500 triggers failover to key_2."""
    session_id = f"fail500-{uuid.uuid4().hex[:8]}"
    res = client.post(
        "/chat",
        json={"session_id": session_id, "message": "Test failover 500"},
        headers={"X-Simulate-Failure": "key_1=500"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["metadata"]["key_id"] == "key_2"
    assert data["metadata"]["fallback_attempts"] == 1


def test_simulate_failure_all_providers_unavailable_503(client):
    """Test X-Simulate-Failure: *=429 causes 503 and writes nothing to SQLite."""
    session_id = f"failall-{uuid.uuid4().hex[:8]}"
    res = client.post(
        "/chat",
        json={"session_id": session_id, "message": "This should fail on all keys"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert res.status_code == 503
    assert res.json() == {"detail": "all providers unavailable"}

    # Failed requests (422 or 503) MUST NOT write anything to SQLite
    assert get_session(session_id) is None
    assert get_messages(session_id) == []
    res_get = client.get(f"/sessions/{session_id}/messages")
    assert res_get.status_code == 404


def test_simulate_failure_flag_disabled(client, monkeypatch):
    """When ENABLE_FAILURE_SIMULATION=false, header is ignored and key_1 succeeds."""
    monkeypatch.setattr(settings, "ENABLE_FAILURE_SIMULATION", False)
    session_id = f"no-sim-{uuid.uuid4().hex[:8]}"

    res = client.post(
        "/chat",
        json={"session_id": session_id, "message": "Simulation disabled test"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["metadata"]["key_id"] == "key_1"
    assert data["metadata"]["fallback_attempts"] == 0
