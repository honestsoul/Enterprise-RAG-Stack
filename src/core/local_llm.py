"""Client for running local transformer models via Hugging Face.

This client loads a model from disk using the Transformers library and
provides generation and embedding capabilities without any network
dependencies.  Because local models can be large, initialising them
may take a while and require significant system resources.  Adjust the
model path in your configuration to point to a directory containing
compatible model weights and a tokenizer.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Sequence

try:
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )
except ImportError:
    AutoModel = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore

import torch

from .base_llm import BaseLLM


class LocalLLM(BaseLLM):
    """Local Hugging Face model client implementing BaseLLM."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "The transformers package is required for LocalLLM. Install it with `pip install transformers`"
            )
        model_path = config.get("model_path") or os.getenv("LOCAL_MODEL_PATH")
        if not model_path:
            raise ValueError(
                "A path to the local model must be provided via 'model_path' in the config or the LOCAL_MODEL_PATH environment variable."
            )
        self.temperature: float = config.get("temperature", 0.7)
        self.max_tokens: int = config.get("max_tokens", 1024)
        # Load the generation model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        # Use the transformers pipeline for text generation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
        # Lazy load an embedding model for feature extraction.  If the
        # underlying generation model supports feature extraction we can
        # reuse it; otherwise we load a separate AutoModel instance.
        try:
            self.embedder_model = AutoModel.from_pretrained(model_path)
        except Exception:
            self.embedder_model = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text locally using the transformers pipeline.

        Args:
            prompt: The input prompt.
            **kwargs: Additional generation keyword arguments accepted by
                transformers pipelines (e.g. `top_p`, `do_sample`).

        Returns:
            The generated completion.
        """
        gen_args = {
            "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        gen_args.update({k: v for k, v in kwargs.items() if k not in {"max_tokens", "temperature"}})
        outputs = self.generator(prompt, **gen_args)
        # The pipeline returns a list of generated sequences; pick the first
        return outputs[0]["generated_text"]

    def embed(self, texts: Sequence[str], **kwargs: Any) -> List[List[float]]:
        """Generate embeddings using the loaded model in feature extraction mode.

        Args:
            texts: A sequence of strings to embed.
            **kwargs: Currently unused.

        Returns:
            A list of embeddings as lists of floats.
        """
        if self.embedder_model is None:
            raise NotImplementedError(
                "The loaded model does not support feature extraction. Use a different model or provider for embeddings."
            )
        embeddings: List[List[float]] = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.embedder_model(**inputs)
            # Use the mean pooling of the last hidden state as a simple embedding
            last_hidden = outputs.last_hidden_state  # type: ignore[attr-defined]
            pooled = last_hidden.mean(dim=1).squeeze().tolist()
            # Convert PyTorch tensor or list to plain Python list of floats
            if isinstance(pooled, torch.Tensor):  # pragma: no cover - type check
                pooled = pooled.detach().cpu().tolist()
            embeddings.append(pooled)  # type: ignore[arg-type]
        return embeddings
