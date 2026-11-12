import pytest

from copilot import llm


def test_make_client_uses_gemini_settings(monkeypatch):
    seen = {}

    class FakeGeminiChatClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(llm, "GeminiChatClient", FakeGeminiChatClient)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_MODEL", raising=False)

    client = llm.make_client()

    assert isinstance(client, FakeGeminiChatClient)
    assert seen == {"api_key": "test-key", "model": "gemini-3.5-flash-lite"}


def test_make_client_requires_google_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Missing GOOGLE_API_KEY"):
        llm.make_client()
