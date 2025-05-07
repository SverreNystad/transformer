from src.embedder import Embedder
import pytest

@pytest.fixture(scope="module")
def model():
    return Embedder()

def test_embed_text(model):

    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium."
    ]
    embeddings = model.encode(sentences)

    assert len(embeddings) == len(sentences)
    assert embeddings[0].shape >= (1,)

