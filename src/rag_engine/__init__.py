"""Hybrid RAG production engine."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hybrid-rag-production-engine")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"
