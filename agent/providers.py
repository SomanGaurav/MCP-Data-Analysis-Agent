"""Unified LLM interface across Ollama / Groq / Gemini via LiteLLM.

Every provider is called through one OpenAI-style ``completion`` so the rest of
the agent never branches on which model is active. Switching provider is just
constructing this with a different name (driven by the UI dropdown or env).
"""

from __future__ import annotations

from typing import Any, Optional

import litellm

from agent.config import ProviderConfig, load_provider

# Keep LiteLLM quiet and resilient to provider-specific extra params.
litellm.drop_params = True
litellm.suppress_debug_info = True


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.cfg: ProviderConfig = load_provider(provider)

    @property
    def provider(self) -> str:
        return self.cfg.provider

    @property
    def model(self) -> str:
        return self.cfg.model

    def is_ready(self) -> tuple[bool, str]:
        """Whether the selected provider can be called; second item explains if not."""
        if self.cfg.needs_key and not self.cfg.api_key:
            return False, (
                f"{self.provider} needs an API key. Set "
                f"{self.provider.upper()}_API_KEY in your .env."
            )
        return True, "ready"

    def complete(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ):
        """Call the model. Returns a LiteLLM (OpenAI-shaped) response object.

        Args:
            messages: OpenAI-style chat messages.
            tools: OpenAI-style tool/function schemas (optional).
            temperature: Sampling temperature.
        """
        params: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.cfg.api_key:
            params["api_key"] = self.cfg.api_key
        if self.cfg.api_base:
            params["api_base"] = self.cfg.api_base
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        params.update(kwargs)
        return litellm.completion(**params)
