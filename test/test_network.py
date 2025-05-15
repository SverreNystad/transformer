import torch

from src.embedder import Embedder, create_vocab_mapping, make_causal_mask
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
    sequence_length = 10

    sentence = "This is a test."
    vocab = create_vocab_mapping([sentence])
    src_sequence = sentence
    target_sequence = sentence
    src_embedding = Embedder(vocab_table=vocab, embedding_dimension=d_model, max_seq_len=sequence_length)
    target_embedding = Embedder(vocab_table=vocab, embedding_dimension=d_model, max_seq_len=sequence_length)
    src_tensor = src_embedding(src_sequence)
    target_tensor = target_embedding(target_sequence)
    casual_mask = make_causal_mask(sequence_length).to(target_tensor.device)
    vocab_size = src_embedding.vocab_size

    transformer = Transformer(
        output_classes=vocab_size,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
    )

    output = transformer(src_tensor, target_tensor, casual_mask)
    assert output.shape == (
        1,
        sequence_length,
        vocab_size,
    ), f"Expected shape (1, {sequence_length}, {vocab_size}), but got {output.shape}"


def test_decoder_only_transformer():
    d_model = 4
    n_heads = 2
    d_ff = 8
    decoder_layers = 2
    encoder_layers = 0
    sequence_length = 10

    sentence = "This is a test."
    vocab = create_vocab_mapping([sentence])
    src_sequence = sentence
    target_sequence = sentence
    src_embedding = Embedder(vocab_table=vocab, embedding_dimension=d_model, max_seq_len=sequence_length)
    target_embedding = Embedder(vocab_table=vocab, embedding_dimension=d_model, max_seq_len=sequence_length)
    src_tensor = src_embedding(src_sequence)
    target_tensor = target_embedding(target_sequence)
    casual_mask = make_causal_mask(sequence_length).to(target_tensor.device)
    vocab_size = src_embedding.vocab_size

    transformer = Transformer(
        output_classes=vocab_size,
        num_encoder_layers=encoder_layers,
        num_decoder_layers=decoder_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
    )

    output = transformer(src_tensor, target_tensor, casual_mask)
    assert output.shape == (
        1,
        sequence_length,
        vocab_size,
    ), f"Expected shape (1, {sequence_length}, {vocab_size}), but got {output.shape}"


def test_encoder_only_transformer():
    d_model = 4
    n_heads = 2
    d_ff = 8
    decoder_layers = 0
    encoder_layers = 2
    sequence_length = 10

    sentence = "This is a test."
    vocab = create_vocab_mapping([sentence])
    src_sequence = sentence
    target_sequence = sentence
    src_embedding = Embedder(vocab_table=vocab, embedding_dimension=d_model, max_seq_len=sequence_length)
    target_embedding = Embedder(vocab_table=vocab, embedding_dimension=d_model, max_seq_len=sequence_length)
    src_tensor = src_embedding(src_sequence)
    target_tensor = target_embedding(target_sequence)
    casual_mask = make_causal_mask(sequence_length).to(target_tensor.device)
    vocab_size = src_embedding.vocab_size

    transformer = Transformer(
        output_classes=vocab_size,
        num_encoder_layers=encoder_layers,
        num_decoder_layers=decoder_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
    )

    output = transformer(src_tensor, target_tensor, casual_mask)
    assert output.shape == (
        1,
        sequence_length,
        vocab_size,
    ), f"Expected shape (1, {sequence_length}, {vocab_size}), but got {output.shape}"
