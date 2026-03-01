"""Factory for constructing LLM clients based on configuration.

The factory centralises the logic for instantiating concrete
implementations of the ``BaseLLM`` interface.  By passing the name of
the provider and a dictionary of configuration values, callers can
obtain a ready‑to‑use model client without having to import specific
subclasses.  To add support for new providers, implement the
``BaseLLM`` interface in a new module and update the mapping below.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore

from .base_llm import BaseLLM
from .gpt_client import GPTClient
from .claude_client import ClaudeClient
from .local_llm import LocalLLM


# Mapping from provider name to the corresponding client class
_PROVIDER_MAP = {
    "openai": GPTClient,
    "anthropic": ClaudeClient,
    "local": LocalLLM,
}


def get_model(provider: str, config: Dict[str, Any] | None = None) -> BaseLLM:
    """Instantiate and return a model client for the given provider.

    Args:
        provider: The key identifying which provider to use (e.g. ``"openai"``).
        config: A dictionary of provider‑specific configuration values.  If
            omitted, the factory will attempt to load configuration from
            ``config/model_config.yaml`` relative to the project root and
            select the appropriate section based on the provider key.

    Returns:
        An instance of ``BaseLLM`` corresponding to the requested provider.

    Raises:
        KeyError: If the provider is unknown.
        FileNotFoundError: If no configuration file can be found.
    """
    provider = provider.lower()
    if provider not in _PROVIDER_MAP:
        raise KeyError(f"Unknown provider: {provider}")

    if config is None:
        # Attempt to load configuration from the YAML file
        config_path_candidates = [
            Path(__file__).resolve().parents[2] / "config" / "model_config.yaml",
            Path(os.getcwd()) / "config" / "model_config.yaml",
        ]
        for path in config_path_candidates:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    full_config = yaml.safe_load(f) or {}
                provider_config = full_config.get("providers", {}).get(provider, {})
                break
        else:
            raise FileNotFoundError(
                "Could not find model_config.yaml in expected locations."
            )
        config = provider_config

    client_cls = _PROVIDER_MAP[provider]
    return client_cls(config)


def get_default_model() -> BaseLLM:
    """Return the model specified as the default provider in the config file.

    This convenience function reads ``config/model_config.yaml`` and
    instantiates the provider indicated by the ``llm_provider`` key.  If
    the configuration file cannot be found or the provider is unknown,
    an exception is raised.
    """
    config_path = Path(__file__).resolve().parents[2] / "config" / "model_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f) or {}
    provider: str = full_config.get("llm_provider")
    return get_model(provider, full_config.get("providers", {}).get(provider))
