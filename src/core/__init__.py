"""Core modules exposing the low level LLM clients and factory.

The `core` package contains the abstractions for communicating with
different large language model providers.  Each client implements the
``BaseLLM`` interface defined in ``base_llm.py``.  The factory in
``model_factory.py`` centralises the instantiation of clients based
on a configuration file, allowing you to switch providers without
modifying call sites.
"""

from .base_llm import BaseLLM  # noqa: F401
from .model_factory import get_model  # noqa: F401
