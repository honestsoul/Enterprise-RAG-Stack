"""Client for interacting with OpenAI’s ChatCompletion and Embedding APIs.

This client wraps the `openai` Python SDK to provide a unified interface
for generating completions and embeddings.  It pulls configuration from
``config/model_config.yaml`` via the model factory and falls back to
environment variables when necessary.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

try:
    import openai  # type: ignore
except ImportError:
    # When the openai package is not installed the client will raise at runtime.
    openai = None  # type: ignore

from .base_llm import BaseLLM


class GPTClient(BaseLLM):
    """OpenAI client implementation of the BaseLLM interface."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        if openai is None:
            raise ImportError(
                "The openai package is not installed. Install it with `pip install openai`"
            )
        # Extract configuration values
        self.model_name: str = config.get("model_name", "gpt-4o")
        self.temperature: float = config.get("temperature", 0.7)
        self.max_tokens: int = config.get("max_tokens", 1024)
        # Set the API key either from the config or the environment
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Call the OpenAI ChatCompletion API to generate a response.

        Args:
            prompt: The user prompt to send to the model.
            **kwargs: Additional parameters such as system_prompt or stop sequences.

        Returns:
            The generated string.
        """
        # Build the message payload for the chat completion API
        system_prompt = kwargs.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        # Extract and return the assistant message content
        return response.choices[0].message["content"]  # type: ignore[index]

    def embed(self, texts: Sequence[str], **kwargs: Any) -> List[List[float]]:
        """Call the OpenAI embedding endpoint for a list of texts.

        Args:
            texts: A sequence of strings to embed.
            **kwargs: Additional parameters such as model name.

        Returns:
            A list of embedding vectors (lists of floats).
        """
        model = kwargs.get("embedding_model", "text-embedding-ada-002")
        response = openai.Embedding.create(input=list(texts), model=model)
        return [e["embedding"] for e in response["data"]]  # type: ignore[index]
