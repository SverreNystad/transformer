import math
from sentence_transformers import SentenceTransformer
import torch
import torch
from torch import nn
from torch import Tensor
from typing import Optional, Tuple

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class Embedder(nn.Module):
    def __init__(self, vocab_size: int, d_in: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_in)
        
        self.positional_embedding   = nn.Embedding(max_seq_len, d_in)
        self.dropout   = nn.Dropout(dropout)

        # Layer norm to stabilize embedding magnitudes
        self.layer_norm = nn.LayerNorm(d_in)


    def forward(self, sentence: str) -> Tensor:
        """
        Args:
            sentence: str
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """

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
        div = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-math.log(wavelength) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of same shape
        """
        return x + self.pe[:, :x.size(1)]

        

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
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium."
    ]
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    similarities = model.similarity(embeddings, embeddings)
    print(similarities.shape)
