from torch import nn
from torch import Tensor
from typing import Optional, Tuple

import torch


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.linear_q = nn.Linear(d_model, n_heads * self.d_k)
        self.linear_k = nn.Linear(d_model, n_heads * self.d_k)
        self.linear_v = nn.Linear(d_model, n_heads * self.d_v)
        self.merger = nn.Linear(n_heads * self.d_v, d_model)
        self.attention = ScaledDotProductAttention()
    
    def forward(
        self, 
        queries: Tensor,  # (batch_size, n_heads, len_q, d_k)
        keys: Tensor,  # (batch_size, n_heads, len_k, d_k)
        values: Tensor,  # (batch_size, n_heads, len_k, d_v)
        ) -> Tensor:

        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(values)

        attention_scores = self.attention(Q, K, V)
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
        mask: Optional[Tensor] = None  # (batch_size, 1, len_q, len_k)
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            attn_output: Tensor of shape (batch_size, n_heads, len_q, d_v)
        """
        
        # Z = Softmax(QK^T / sqrt(d_k))*V
        # Q: Queries, K: Keys, V: Values
        d_k = keys.size(-1)
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = self.softmax(attention_scores)
        attn_output = torch.matmul(attention_weights, values)
        return attn_output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of same shape
        """
        pass