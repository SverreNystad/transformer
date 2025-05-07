from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    """
    A class to take in a sequence of words and transform them into a vector representation.
    This Vector representation will embed the token and combine the positional information of the token in the sentence.
    """
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences) -> torch.Tensor:
        return self.model.encode(sentences)

    def similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor):
        return self.model.similarity(embeddings1, embeddings2)

if __name__ == "__main__":
    model = Embedder()
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium."
    ]
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    similarities = model.similarity(embeddings, embeddings)
    print(similarities.shape)
