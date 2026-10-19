"""Tests that download model files from Hugging Face. Run with `pytest -m models`."""

import numpy as np
import pytest

pytestmark = pytest.mark.models


def test_bm25_encoder_drops_stop_words_and_weights_repeated_terms_higher() -> None:
    from rag_engine.indexing.sparse import Bm25SparseEncoder

    encoder = Bm25SparseEncoder(avg_len=10)
    [vector] = encoder.encode_documents(["The rollout of the rollout plan"])
    query = encoder.encode_query("rollout plan")

    assert len(vector.indices) == len(vector.values) == 2  # "the" and "of" are stop words
    assert set(query.indices) == set(vector.indices)
    assert max(vector.values) > min(vector.values)  # "rollout" appears twice, "plan" once


def test_the_mean_document_length_counts_terms_after_stop_words() -> None:
    from rag_engine.indexing.sparse import mean_document_length

    # 3 terms (rollout, rollout, plan) and 4 terms (alpha, beta, gamma, delta).
    texts = ["The rollout of the rollout plan", "alpha beta gamma delta"]

    assert mean_document_length(texts) == pytest.approx(3.5, abs=0.01)


def test_dense_embedder_returns_unit_vectors_and_distinguishes_queries() -> None:
    pytest.importorskip("sentence_transformers")
    from rag_engine.indexing.embedders import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(device="cpu", batch_size=4)
    documents = embedder.embed_documents(
        ["Rotate the API keys every ninety days.", "The cafeteria menu changes on Mondays."]
    )
    query = embedder.embed_query("how often should API keys be rotated")

    assert documents.shape == (2, 768)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1, atol=1e-4)
    assert float(query @ documents[0]) > float(query @ documents[1])
