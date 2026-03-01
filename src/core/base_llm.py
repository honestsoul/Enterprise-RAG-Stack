"""Abstract base class for Large Language Model clients.

All concrete model clients must implement this interface.  The
``generate`` method returns a completion given a prompt, and the
``embed`` method returns a vector representation of the input text
useful for similarity search.  Additional provider‑specific methods
should be encapsulated in the subclass rather than leaking into
higher levels of the application.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Sequence


class BaseLLM(ABC):
    """Abstract base class for Large Language Model (LLM) clients.

    Subclasses must implement the ``generate`` and ``embed`` methods.  The
    constructor accepts a dictionary of configuration values specific to
    the provider.  It should be idempotent and avoid heavy initialisation
    until required.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input text to send to the model.
            **kwargs: Additional provider‑specific keyword arguments.

        Returns:
            The model’s response as a string.
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: Sequence[str], **kwargs: Any) -> List[List[float]]:
        """Generate embeddings for a sequence of texts.

        Args:
            texts: A sequence of strings to embed.
            **kwargs: Additional provider‑specific keyword arguments.

        Returns:
            A list of embeddings where each embedding is a list of floats.
        """
        raise NotImplementedError
