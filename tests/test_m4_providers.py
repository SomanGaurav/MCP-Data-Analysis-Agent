"""M4 tests: provider switch resolution (no live LLM calls)."""

import pytest

from agent.config import load_provider
from agent.providers import LLMClient


def test_ollama_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    cfg = load_provider("ollama")
    assert cfg.model == "ollama_chat/qwen2.5:3b"
    assert cfg.api_base and cfg.is_configured   # no key needed


def test_groq_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = load_provider("groq")
    assert cfg.model.startswith("groq/") and cfg.needs_key
    assert not cfg.is_configured

    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    assert load_provider("groq").is_configured


def test_gemini_resolution(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    cfg = load_provider("gemini")
    assert cfg.model == "gemini/gemini-2.0-flash" and cfg.needs_key


def test_invalid_provider():
    with pytest.raises(ValueError):
        load_provider("openai")


def test_client_readiness(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    ready, msg = LLMClient("groq").is_ready()
    assert not ready and "API key" in msg

    ready, _ = LLMClient("ollama").is_ready()
    assert ready
