"""Exceptions that cross module boundaries."""


class LLMError(Exception):
    """A model call failed for good: bad credentials, a rejected request, or retries ran out."""


class BudgetExceededError(Exception):
    """The money set aside for a run is spent."""


class RetrievalUnavailableError(Exception):
    """Retrieval cannot run at all, for example because a model or a data service cannot load."""


class GenerationUnavailableError(Exception):
    """Answers cannot be generated at all, for example because no model credentials are set."""
