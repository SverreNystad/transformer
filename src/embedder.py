import math

import spacy
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn


class Embedder(nn.Module):
    def __init__(self, embedding_dimension: int, max_seq_len: int) -> None:
        super().__init__()
        # Load the Norwegian language model
        self.max_seq_len = max_seq_len
        self.nlp = spacy.load("nb_core_news_sm")
        vocab_size = len(self.nlp.vocab)
        print(f"Vocab size: {vocab_size}")
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dimension)

        self.positional_embedding = PositionalEncoding(d_model=embedding_dimension, max_len=max_seq_len)

        # Layer norm to stabilize embedding magnitudes
        self.layer_norm = nn.LayerNorm(embedding_dimension)

    def forward(self, sentence: str) -> Tensor:
        """
        Args:
            sentence: str
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Tokenize the sentence
        doc = self.nlp(sentence)
        token_ids = [token.idx for token in doc]
        if len(token_ids) < self.max_seq_len:
            # Pad the sequence with zeros
            token_ids += [0] * (self.max_seq_len - len(token_ids))

        # Convert to tensor (batch_size, seq_len)
        token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)

        token_embeddings = self.token_embedding(token_ids_tensor)
        positional_embeddings = self.positional_embedding(token_embeddings)
        embeddings = self.layer_norm(positional_embeddings)
        return embeddings


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer models.
    Using the Sinusoidal Positional Encoding as described in the original Transformer paper. "Attention is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        wavelength = 10000
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(wavelength) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)  # Even indices
        pe[:, 1::2] = torch.cos(pos * div)  # Odd indices
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of same shape
        """
        return x + self.pe[:, : x.size(1)]


def make_causal_mask(size: int) -> Tensor:
    """
    Create a causal mask for the decoder to prevent attending to future tokens.
    """
    # triu with diagonal=1 gives 1s above diagonal; invert to get causal
    mask = ~torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


class ExternalEmbedder(nn.Module):
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
    model = ExternalEmbedder()
    sentences = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    similarities = model.similarity(embeddings, embeddings)
    print(similarities.shape)
