import numpy as np

from fast_sentence_transformers import FastSentenceTransformer
from fast_sentence_transformers.fast_sentence_transformers import (
    _DEFAULT_EMBEDDING_MODEL,
)

text = "Hello hello, hey, hello hello"


def test_standalone():
    fast_transformer = FastSentenceTransformer(_DEFAULT_EMBEDDING_MODEL)
    embeddings = fast_transformer.encode(text)
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings) == 384
    embeddings = fast_transformer.encode([text, text])
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings) == 2
