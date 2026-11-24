import os
import tempfile
import uuid
import pytest
from fastapi.testclient import TestClient

# Configure environment before app imports
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["LLM_API_KEYS"] = "key_1,key_2,key_3"
os.environ["ENABLE_FAILURE_SIMULATION"] = "true"
os.environ["LLM_MODEL"] = "gemini/gemini-2.0-flash"
os.environ["DEFAULT_SYSTEM_PROMPT"] = "You are a helpful AI assistant."


class MockMessage:
    def __init__(self, content: str):
        self.content = content


class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockModelResponse:
    def __init__(
        self,
        content: str = "Test LLM reply",
        model: str = "gemini/gemini-2.0-flash",
        prompt_tokens: int = 14,
        completion_tokens: int = 8,
    ):
        self.choices = [MockChoice(content)]
        self.model = model
        self.usage = MockUsage(prompt_tokens, completion_tokens)
        self._hidden_params = {}


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Sets up an isolated SQLite database and local cache for each test."""
    temp_db_path = os.path.join(tempfile.gettempdir(), f"test_db_{uuid.uuid4().hex}.db")
    test_db_url = f"sqlite:///{temp_db_path}"
    from app.config import settings
    monkeypatch.setattr(settings, "DATABASE_URL", test_db_url)
    monkeypatch.setattr(settings, "LLM_API_KEYS", "key_1,key_2,key_3")
    monkeypatch.setattr(settings, "ENABLE_FAILURE_SIMULATION", True)

    from app import database, llm
    database.init_db()
    llm._LOCAL_COMPLETION_CACHE.clear()

    # Default mock for litellm.completion
    call_count = [0]

    def mock_completion(*args, **kwargs):
        call_count[0] += 1
        messages = kwargs.get("messages", [])
        last_user_msg = "test response"
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "test response")
                break
        tokens_in = len(last_user_msg.split()) * 2 + 5
        tokens_out = 8
        return MockModelResponse(
            content=f"Echo: {last_user_msg}",
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
        )

    monkeypatch.setattr(llm.litellm, "completion", mock_completion)

    yield

    try:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
    except Exception:
        pass


@pytest.fixture
def client():
    from app.main import app
    client = TestClient(app, raise_server_exceptions=False)
    yield client
