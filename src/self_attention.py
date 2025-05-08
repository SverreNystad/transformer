import torch
from torch import Tensor, nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask of shape (batch_size, 1, seq_len, seq_len)
        Returns:
            Tensor of same shape
        """
        attention = self.attention(queries, keys, values, mask)
        attention = self.dropout1(attention)
        # Add & Norm
        x = queries + keys + values
        x = self.layer_norm1(x + attention)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        # Residual connection
        output = self.layer_norm2(x + ffn_output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.linear_q = nn.Linear(d_model, n_heads * self.d_k)  # W_q
        self.linear_k = nn.Linear(d_model, n_heads * self.d_k)  # W_k
        self.linear_v = nn.Linear(d_model, n_heads * self.d_v)  # W_v
        self.merger = nn.Linear(n_heads * self.d_v, d_model)  # W_o
        self.attention = ScaledDotProductAttention()

    def forward(
        self,
        queries: Tensor,  # (batch_size, n_heads, len_q, d_k)
        keys: Tensor,  # (batch_size, n_heads, len_k, d_k)
        values: Tensor,  # (batch_size, n_heads, len_k, d_v)
        mask: Tensor | None = None,  # (batch_size, 1, len_q, len_k)
    ) -> Tensor:
        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(values)

        attention_scores = self.attention(Q, K, V, mask)
        merged_attention = self.merger(attention_scores)
        return merged_attention


class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        queries: Tensor,  # (batch_size, n_heads, len_q, d_k)
        keys: Tensor,  # (batch_size, n_heads, len_k, d_k)
        values: Tensor,  # (batch_size, n_heads, len_k, d_v)
        mask: Tensor | None = None,  # (batch_size, 1, len_q, len_k)
    ) -> Tensor:
        """
        Returns:
            attn_output: Tensor of shape (batch_size, n_heads, len_q, d_v)
        """

        # Z = Softmax(QK^T / sqrt(d_k))*V
        # Q: Queries, K: Keys, V: Values
        d_k = keys.size(-1)
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (d_k**0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = self.softmax(attention_scores)
        attn_output = torch.matmul(attention_weights, values)
        return attn_output


class PositionwiseFeedForward(nn.Module):
    """
    The Feed Forward Network (FFN) in the Transformer architecture.
    It consists of an expand-and-contract structure

    Mathematically, it can be represented as: FFN(x) = max(0, xW1 + b1)W2 + b2
    Or as the paper states: "Another way of describing this is as two convolutions with kernel size 1" (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.linear1(x)
        logits = self.relu(logits)
        logits = self.linear2(logits)
        logits = self.dropout(logits)
        return logits
