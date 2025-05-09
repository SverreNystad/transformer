import torch

from src.network import Decoder, Encoder, Transformer


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


def test_decoder():
    d_model = 4
    n_heads = 2
    d_ff = 8
    num_layers = 2
    sequence_length = 6
    decoder_block = Decoder(num_layers=num_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff)

    target_embedding = torch.randn(1, sequence_length, d_model)
    memory_embedding = torch.randn(1, sequence_length, d_model)

    output = decoder_block(target_embedding, memory_embedding)
    assert output.shape == (
        1,
        sequence_length,
        d_model,
    ), f"Expected shape (1, {sequence_length}, {d_model}), but got {output.shape}"


def test_transformer():
    d_model = 4
    n_heads = 2
    d_ff = 8
    num_layers = 2
    sequence_length = 6
    transformer = Transformer(
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=sequence_length,
    )

    vocab_size = transformer.vocab_size

    src_sequence = "This is a test."
    target_sequence = "This is a test."

    output = transformer(src_sequence, target_sequence)
    assert output.shape == (
        1,
        sequence_length,
        vocab_size,
    ), f"Expected shape (1, {sequence_length}, {vocab_size}), but got {output.shape}"
