import torch

from src.self_attention import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    ScaledDotProductAttention,
    TransformerBlock,
)


def test_scaled_dot_product_attention():
    d_out = 2
    sequence_length = 6
    attention_block = ScaledDotProductAttention()
    queries = torch.randn(1, sequence_length, d_out)
    keys = torch.randn(1, sequence_length, d_out)
    values = torch.randn(1, sequence_length, d_out)

    self_attention = attention_block(queries, keys, values)
    assert self_attention.shape == (
        1,
        sequence_length,
        d_out,
    ), f"Expected shape (1, {sequence_length}, {d_out}), but got {self_attention.shape}"


def test_multi_head_attention():
    d_model = 4
    n_heads = 2
    sequence_length = 6
    attention_block = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    queries = torch.randn(1, sequence_length, d_model)
    keys = torch.randn(1, sequence_length, d_model)
    values = torch.randn(1, sequence_length, d_model)

    self_attention = attention_block(queries, keys, values)
    assert self_attention.shape == (
        1,
        sequence_length,
        d_model,
    ), f"Expected shape (1, {sequence_length}, {d_model}), but got {self_attention.shape}"


def test_positionwise_feed_forward():
    d_model = 4
    d_ff = 8
    sequence_length = 6
    feed_forward_block = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

    x = torch.randn(1, sequence_length, d_model)

    output = feed_forward_block(x)
    assert output.shape == (
        1,
        sequence_length,
        d_model,
    ), f"Expected shape (1, {sequence_length}, {d_model}), but got {output.shape}"


def test_transformer_block():
    d_model = 4
    n_heads = 2
    d_ff = 8
    sequence_length = 6
    transformer_block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

    queries = torch.randn(1, sequence_length, d_model)
    keys = torch.randn(1, sequence_length, d_model)
    values = torch.randn(1, sequence_length, d_model)

    output = transformer_block(queries, keys, values)
    assert output.shape == (
        1,
        sequence_length,
        d_model,
    ), f"Expected shape (1, {sequence_length}, {d_model}), but got {output.shape}"
