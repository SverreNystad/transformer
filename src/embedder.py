import math

import nltk
import torch
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn

nltk.download("punkt_tab")

START_TOKEN = "<START>"
PADDING_TOKEN = "<PADDING>"
END_TOKEN = "<END>"
UNKNOWN_TOKEN = "<UNKNOWN>"


def create_vocab_mapping(raw_sentences: list[str], language="english") -> dict[str, int]:
    """
    Create a vocabulary mapping from raw sentences.
    The mapping will include special tokens for start, padding, end, and unknown tokens.
    """
    vocab = {START_TOKEN: 0, PADDING_TOKEN: 1, END_TOKEN: 2, UNKNOWN_TOKEN: 3}
    i = len(vocab)
    for sentence in raw_sentences:
        tokens = word_tokenize(sentence, language=language)
        for token in tokens:
            if token not in vocab:
                vocab[token] = i
                i += 1
    return vocab


class Embedder(nn.Module):
    def __init__(self, vocab_table: dict[str, int], embedding_dimension: int, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_table = vocab_table
        self.vocab_size = len(vocab_table)
        self.token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dimension)

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

        token_ids = self.get_token_ids(sentence)
        # Convert to tensor (batch_size, seq_len)
        token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(self.token_embedding.weight.device)

        return self.embed_ids(token_ids_tensor)

    def embed_ids(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Given token_ids of shape (batch_size, seq_len),
        return the full embeddings (batch_size, seq_len, d_model)
        with positional embedding + layer norm applied.
        """
        x = self.token_embedding(token_ids)
        x = self.positional_embedding(x)
        return self.layer_norm(x)

    def get_token_ids(self, sentence: str, language="english") -> list[int]:
        """
        Convert a sentence to token IDs using the vocabulary mapping.
        """
        # Tokenize the sentence
        tokens = word_tokenize(sentence, language=language)
        # Convert tokens to token IDs using the vocabulary mapping
        token_ids = [self.vocab_table.get(token, self.vocab_table[UNKNOWN_TOKEN]) for token in tokens]

        # Pad the token IDs to the maximum sequence length
        max_seq_len = self.max_seq_len - 2  # Account for start and end tokens
        if len(token_ids) < max_seq_len:
            token_ids += [self.vocab_table[PADDING_TOKEN]] * (max_seq_len - len(token_ids))

        # Add start and end tokens
        token_ids = [self.vocab_table[START_TOKEN]] + token_ids + [self.vocab_table[END_TOKEN]]
        return token_ids

    def get_token(self, token_id: int) -> str:
        """
        Convert a token ID to the corresponding token using the vocabulary mapping.
        """
        for token, idx in self.vocab_table.items():
            if idx == token_id:
                return token
        return UNKNOWN_TOKEN


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


def make_causal_mask(size: int, device="cpu") -> Tensor:
    """
    Create a causal mask for the decoder to prevent attending to future tokens.
    """
    # triu with diagonal=1 gives 1s above diagonal; invert to get causal
    mask = ~torch.triu(torch.ones(size, size), diagonal=1).bool().to(device)
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
