import torch
from torch import Tensor, nn

from src.embedder import Embedder
from src.self_attention import TransformerBlock


class Encoder(nn.Module):
    """
    The Encoder contains a Self-attention block (layer) that computes the relationship between different words in a sequence, as well as a Feedforward block.
    It also has a Residual skip connections around the blocks along with normalisation layers after each block.
    """

    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            src: Tensor of shape (batch_size, src_len, d_model)
            src_mask: Optional mask tensor
        Returns:
            Tensor of shape (batch_size, src_len, d_model)
        """
        for layer in self.layers:
            src = layer(src, src, src, src_mask)
        return src


class Decoder(nn.Module):
    """
    The Decoder also contains the Self-attention block and the Feedforward block, as well as a second Encoder-Decoder attention block.
    It also has a Residual skip connections around the blocks along with normalisation layers after each block.
    """

    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.self_attn_layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout, include_ffn=False) for _ in range(num_layers)]
        )
        self.cross_attn_ffn_layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, target: Tensor, memory: Tensor, target_mask: Tensor | None = None, memory_mask: Tensor | None = None
    ) -> Tensor:
        """
        Compute the decoder's forward pass over a batch of inputs.

        Args:
            target (Tensor): Input embeddings for the decoder of shape (batch_size, tgt_len, d_model).
            memory (Tensor): Encoder output embeddings (“memory”) of shape (batch_size, src_len, d_model).
            target_mask (Optional[Tensor]): Optional mask for target-target attention (e.g., look-ahead mask), shape (tgt_len, tgt_len).
            memory_mask (Optional[Tensor]): Optional mask for target-memory attention, shape (tgt_len, src_len).

        Returns:
            Tensor: Decoder outputs of shape (batch_size, tgt_len, d_model).
        """

        for decoder_block, encoder_decoder_block in zip(self.self_attn_layers, self.cross_attn_ffn_layers):
            # Compute the relevance of each token in the target sequence to each other token in the target sequence.
            target = decoder_block(target, target, target, target_mask)
            # In the Encoder-Decoder Attention, the Query is obtained from the target sequence and the Key/Value from the source sentence.
            # Thus it computes the relevance of each token in the target sequence to each token in the source sequence.
            target = encoder_decoder_block(target, memory, memory, memory_mask)
        return target


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.embedder = Embedder(embedding_dimension=d_model, max_seq_len=max_seq_len)
        self.encoder = Encoder(num_layers=num_encoder_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.decoder = Decoder(num_layers=num_decoder_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.max_seq_len = max_seq_len
        self.vocab_size = self.embedder.token_embedding.num_embeddings

        # Final linear layer to project the decoder output to the vocabulary size
        self.linear = nn.Linear(d_model, self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        src_sequence: str,
        target_sequence: str,
    ) -> Tensor:
        """
        Returns:
            logits: Tensor of shape (batch_size, tgt_len, vocab_size)
        """
        # Encode the source sequence
        src_embedding: Tensor = self.embedder(src_sequence)
        memory: Tensor = self.encoder(src_embedding)

        # Decode the target sequence
        causal_mask = self._make_causal_mask(self.max_seq_len).to(src_embedding.device)  # True on diagonal & below, False above
        target_embedding: Tensor = self.embedder(target_sequence)
        output: Tensor = self.decoder(target_embedding, memory, causal_mask)

        # Project to vocabulary size
        logits = self.linear(output)
        return logits

    def _make_causal_mask(self, size: int) -> Tensor:
        """
        Create a causal mask for the decoder to prevent attending to future tokens.
        """
        # triu with diagonal=1 gives 1s above diagonal; invert to get causal
        mask = ~torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
