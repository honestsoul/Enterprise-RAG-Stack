"""Define and render Jinja2 prompt templates for the RAG pipeline.

Templates are simple text files with placeholders that are filled
using Jinja2’s rendering engine.  You can register additional
templates by extending the ``TEMPLATES`` dictionary.  Use the
``PromptTemplate`` class to encapsulate a template’s name and
contents and to provide a convenience ``render`` method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from jinja2 import Template


@dataclass
class PromptTemplate:
    """Container for a prompt template.

    Attributes:
        name: Logical name for the template.
        content: The raw Jinja2 template string.
    """

    name: str
    content: str

    def render(self, **kwargs: Any) -> str:
        """Render the template with the supplied keyword arguments.

        Args:
            **kwargs: Variables available to the template.

        Returns:
            The rendered prompt string.
        """
        template = Template(self.content)
        return template.render(**kwargs)


# Built‑in templates.  Add new templates here as your application grows.
TEMPLATES: Dict[str, PromptTemplate] = {
    "qa": PromptTemplate(
        name="qa",
        content=(
            "You are an assistant answering a user’s question based on the provided context.\n"
            "Answer the question concisely and cite sources by index in square brackets.\n\n"
            "Context:\n{{ context }}\n\n"
            "Question: {{ question }}\n\n"
            "Answer:"
        ),
    ),
    "summarise": PromptTemplate(
        name="summarise",
        content=(
            "Summarise the following content into a concise paragraph:\n\n{{ content }}"
        ),
    ),
}


def get_template(name: str) -> PromptTemplate:
    """Fetch a registered template by name.

    Args:
        name: Name of the template (key in ``TEMPLATES``).

    Returns:
        The corresponding ``PromptTemplate``.

    Raises:
        KeyError: If the template name is unknown.
    """
    try:
        return TEMPLATES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown template name: {name}") from exc
