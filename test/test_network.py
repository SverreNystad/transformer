import torch

from src.network import Encoder


def test_encoder():
    d_model = 4
    n_heads = 2
    d_ff = 8
    num_layers = 2
    sequence_length = 6
    encoder_block = Encoder(num_layers=num_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff)

    embedding = torch.randn(1, sequence_length, d_model)

    output = encoder_block(embedding)
    assert output.shape == (
        1,
        sequence_length,
        d_model,
    ), f"Expected shape (1, {sequence_length}, {d_model}), but got {output.shape}"
