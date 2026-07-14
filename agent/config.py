"""Agent configuration: provider selection and model settings from env/.env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

VALID_PROVIDERS = ("ollama", "groq", "gemini")


@dataclass
class ProviderConfig:
    provider: str
    model: str          # LiteLLM model string, e.g. "groq/llama-3.3-70b-versatile"
    api_key: str | None
    api_base: str | None

    @property
    def needs_key(self) -> bool:
        return self.provider in ("groq", "gemini")

    @property
    def is_configured(self) -> bool:
        """True if this provider can actually be called (key present when required)."""
        return bool(self.api_key) if self.needs_key else True


def load_provider(provider: str | None = None) -> ProviderConfig:
    """Resolve provider config, letting an explicit arg (the UI switch) win over env."""
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {VALID_PROVIDERS}, got '{provider}'")

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        # ollama_chat/ gives LiteLLM the best tool-calling support for Ollama.
        return ProviderConfig("ollama", f"ollama_chat/{model}", None,
                              os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    if provider == "groq":
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ProviderConfig("groq", f"groq/{model}", os.getenv("GROQ_API_KEY"), None)
    # gemini
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return ProviderConfig("gemini", f"gemini/{model}", os.getenv("GEMINI_API_KEY"), None)
