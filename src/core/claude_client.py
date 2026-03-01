"""Client for interacting with Anthropic’s Claude API.

Anthropic’s API currently focuses on chat and completion models and does
not provide a first‑party embedding endpoint at the time of writing.
Therefore, the embedding method on this client will raise a
``NotImplementedError``.  To use embeddings with Anthropic, combine
this client with an external embedding service such as OpenAI or a
local model.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # type: ignore

from .base_llm import BaseLLM


class ClaudeClient(BaseLLM):
    """Anthropic Claude client implementation of the BaseLLM interface."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        if anthropic is None:
            raise ImportError(
                "The anthropic package is not installed. Install it with `pip install anthropic`"
            )
        self.model_name: str = config.get("model_name", "claude-3")
        self.temperature: float = config.get("temperature", 0.7)
        self.max_tokens: int = config.get("max_tokens", 1024)
        api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion using the Anthropic API.

        Args:
            prompt: The user prompt to send to the model.
            **kwargs: Additional parameters such as system prompt or stop sequences.

        Returns:
            The generated string.
        """
        system_prompt = kwargs.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.content  # type: ignore[attr-defined]

    def embed(self, texts: Sequence[str], **kwargs: Any) -> List[List[float]]:
        """Anthropic does not currently provide embeddings; raise NotImplementedError."""
        raise NotImplementedError(
            "Anthropic’s Claude API does not expose an embedding endpoint. Use a different provider."
        )
