import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from fast_sentence_transformers import FastSentenceTransformer

text = "Hello hello, hey, hello hello"


@pytest.fixture
def standalone():
    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    slow_transformer = SentenceTransformer(model, cache_folder="tmp/model1")
    fast_transformer = FastSentenceTransformer(model, quantize=False, cache_folder="tmp")
    return (slow_transformer, fast_transformer)


def test_standalone(standalone):
    slow_transformer, fast_transformer = standalone
    assert all(np.isclose(slow_transformer.encode(text), fast_transformer.encode(text), rtol=1.0e-3, atol=1.0e-6))
    fast_transformer.encode([text] * 2)


@pytest.fixture
def standalone_quantize():
    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    slow_transformer = SentenceTransformer(model, cache_folder="tmp")
    fast_transformer = FastSentenceTransformer(model, quantize=True, cache_folder="tmp/model2")
    return (slow_transformer, fast_transformer)


def test_quantize(standalone_quantize):
    slow_transformer, fast_transformer = standalone_quantize
    assert all(np.isclose(slow_transformer.encode(text), fast_transformer.encode(text), rtol=3.0e-1, atol=3.0e-1))
    fast_transformer.encode([text] * 2)


@pytest.fixture
def standalone_automodel():
    model = "prajjwal1/bert-tiny"
    slow_transformer = SentenceTransformer(model, cache_folder="tmp")
    fast_transformer = FastSentenceTransformer(model, quantize=True, cache_folder="tmp/model3")
    return (slow_transformer, fast_transformer)


def test_standalone_automodel(standalone_automodel):
    slow_transformer, fast_transformer = standalone_automodel
    assert all(np.isclose(slow_transformer.encode(text), fast_transformer.encode(text), rtol=3.0e-1, atol=3.0e-1))
    fast_transformer.encode([text] * 2)


@pytest.fixture
def standalone_extras():
    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fast_transformer = FastSentenceTransformer(model, quantize=False, cache_folder="tmp")
    return fast_transformer


def test_standalone_extras(standalone_extras):
    standalone_extras(text, output_value=None)
    standalone_extras(text, output_value="token_embeddings")
    standalone_extras(text, output_value="sentence_embedding")
