"""Exceptions that cross module boundaries."""


class RetrievalUnavailableError(Exception):
    """Retrieval cannot run at all, for example because a model or a data service cannot load."""
