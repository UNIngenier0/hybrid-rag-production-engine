"""Cross-encoder reranker. The `embeddings` dependency group provides sentence-transformers."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a sentence-transformers cross-encoder.

    A bi-encoder compares two independent vectors; a cross-encoder reads the query and the passage
    together, which is slower but much better at telling a relevant passage from one that only
    looks similar. It is only run on a short candidate list.
    """

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        from rag_engine.indexing.embedders import resolve_device

        self.device = resolve_device(device, torch)
        self._model = CrossEncoder(
            model_name, revision=revision, device=self.device, max_length=max_length
        )
        if self.device != "cpu":
            self._model.model.half()  # scoring does not need full precision on a GPU
        self._batch_size = batch_size
        self.model_name = model_name

    def score(self, query: str, passages: Sequence[str]) -> NDArray[np.float32]:
        if not passages:
            return np.empty(0, dtype=np.float32)
        scores: Any = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)
