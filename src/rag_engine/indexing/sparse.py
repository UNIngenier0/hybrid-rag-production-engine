"""BM25 sparse encoding, backed by FastEmbed's `Qdrant/bm25` implementation."""

from collections.abc import Sequence

from fastembed import SparseTextEmbedding

from rag_engine.core.models import SparseVector

MODEL_NAME = "Qdrant/bm25"


class Bm25SparseEncoder:
    """Term-frequency weights with BM25 length normalization; Qdrant applies IDF at query time.

    Tokenization (lowercasing, stop-word removal, stemming) is deterministic and has no
    vocabulary: terms are hashed to indices, so the encoder never needs to be fitted on the
    corpus and documents can be encoded in any order.
    """

    model_name = MODEL_NAME

    def __init__(
        self,
        *,
        avg_len: float = 101.0,
        k: float = 1.2,
        b: float = 0.75,
        parallel: int | None = None,
    ) -> None:
        self._model = SparseTextEmbedding(MODEL_NAME, avg_len=avg_len, k=k, b=b)
        self._parallel = parallel

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        embeddings = self._model.embed(list(texts), parallel=self._parallel)
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in embeddings
        ]

    def encode_query(self, text: str) -> SparseVector:
        embedding = next(iter(self._model.query_embed(text)))
        return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


def mean_document_length(texts: Sequence[str], *, k: float = 1.2) -> float:
    """Mean BM25 document length in terms (after stop-word removal and stemming).

    This is the value to give the encoder as `avg_len`. It is measured through the public API: with
    length normalization off (`b=0`) a term seen `n` times gets the weight `n(k+1)/(n+k)`, which
    can be inverted to recover `n`, and the sum of the counts is the document length.
    """
    encoder = Bm25SparseEncoder(b=0.0, k=k)
    lengths = [
        sum(weight * k / (k + 1 - weight) for weight in vector.values)
        for vector in encoder.encode_documents(texts)
    ]
    return sum(lengths) / len(lengths)
