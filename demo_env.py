"""Runtime defaults for local demo commands."""

import os


def configure_openmp() -> None:
    """Allow FAISS/Ollama dependencies to coexist on local Windows demos."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
