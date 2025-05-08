
import torch
from src.self_attention import ScaledDotProductAttention


def test_scaled_dot_product_attention():
    d_out = 2
    sequence_length = 6
    attention_block = ScaledDotProductAttention()
    queries = torch.randn(1, sequence_length, d_out)
    keys = torch.randn(1, sequence_length, d_out)
    values = torch.randn(1, sequence_length, d_out)

    self_attention = attention_block(queries, keys, values)
    assert self_attention.shape == (1, sequence_length, d_out), (
        f"Expected shape (1, {sequence_length}, {d_out}), but got {self_attention.shape}"
    )
