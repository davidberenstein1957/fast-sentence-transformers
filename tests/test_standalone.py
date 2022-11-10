import pytest

from fast_sentence_transformers import FastSentenceTransformer


@pytest.fixture
def standalone():
    return FastSentenceTransformer("all-MiniLM-L6-v2", quantize=False, cache_folder="models")


def test_standalone(standalone):
    standalone.encode("Hello hello, hey, hello hello")
    standalone.encode(["Life is too short to eat bad food!"] * 2)


@pytest.fixture
def standalone_quantize():
    return FastSentenceTransformer("all-MiniLM-L6-v2", quantize=True, cache_folder="tmp")


def test_quantize(standalone_quantize):
    standalone_quantize.encode("Hello hello, hey, hello hello")
    standalone_quantize.encode(["Life is too short to eat bad food!"] * 2)
