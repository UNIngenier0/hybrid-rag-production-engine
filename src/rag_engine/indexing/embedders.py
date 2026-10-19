"""Dense embedders. The `embeddings` dependency group provides sentence-transformers and torch."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Device = Literal["auto", "cpu", "mps", "cuda"]

DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-m-v1.5"
# The exact model snapshot the index is built with, so a rebuild gives identical vectors.
DEFAULT_REVISION = "e58a8f756156a1293d763f17e3aae643474e9b8a"
DEFAULT_DIMENSION = 768

# Query-side instructions that some retrieval models were trained with. Passages get no prefix.
_RETRIEVAL_INSTRUCTION = "Represent this sentence for searching relevant passages: "
QUERY_PREFIXES = {
    "Snowflake/snowflake-arctic-embed-m-v1.5": _RETRIEVAL_INSTRUCTION,
    "snowflake/snowflake-arctic-embed-xs": _RETRIEVAL_INSTRUCTION,
    "BAAI/bge-small-en-v1.5": _RETRIEVAL_INSTRUCTION,
}


class SentenceTransformerEmbedder:
    """Runs a sentence-transformers model on the best available device.

    Embedding dominates ingestion cost, so the device matters: a GPU runs the same model orders of
    magnitude faster than the CPU cores of a laptop. Half precision is used on any accelerator.
    Vectors are L2-normalized.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        revision: str | None = DEFAULT_REVISION,
        device: Device | str = "auto",
        batch_size: int = 128,
        max_seq_length: int = 512,
        query_prefix: str | None = None,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        resolved = resolve_device(device, torch)
        self._model = SentenceTransformer(model_name, revision=revision, device=resolved)
        if resolved != "cpu":
            self._model.half()
        self._model.max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._query_prefix = (
            QUERY_PREFIXES.get(model_name, "") if query_prefix is None else query_prefix
        )
        self.model_name = model_name
        self.revision = revision
        self.device = resolved
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"{model_name} does not report an embedding dimension")
        self.dimension = int(dimension)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vector: NDArray[np.float32] = self._encode([self._query_prefix + text])[0]
        return vector

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        matrix: Any = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(matrix, dtype=np.float32)


def resolve_device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
