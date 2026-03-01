"""Top level package for the generative AI project.

This package exposes high level modules such as `core`, `rag`,
`processing`, `prompts` and `inference`.  The code is intentionally
modular to allow swapping individual components without modifying
consuming code.  See the README in the docs directory for an overview
of how the pieces fit together.
"""

from importlib import metadata

__all__ = ["__version__"]

try:
    # Use importlib.metadata (Python >=3.8) to fetch package version
    __version__ = metadata.version("generative_ai_project")
except metadata.PackageNotFoundError:
    # Fallback when the package has not been installed yet
    __version__ = "0.0.0"
