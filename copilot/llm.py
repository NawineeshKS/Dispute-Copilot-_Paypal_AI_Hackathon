"""Gemini chat client shared by the Writer and Verifier agents."""

from agent_framework.gemini import GeminiChatClient

from copilot.settings import setting


def make_client() -> GeminiChatClient:
    api_key = setting("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in .env")
    return GeminiChatClient(
        api_key=api_key,
        model=setting("GOOGLE_MODEL", "gemini-3.5-flash-lite"),
    )
