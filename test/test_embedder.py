import torch
from src.embedder import Embedder, PositionalEncoding



def test_custom_embedder():

    d_model = 512
    max_len = 50
    model = Embedder(embedding_dimension=d_model, max_seq_len=max_len)
    sentence = "The weather is lovely today."
    embeddings = model(sentence)
    assert embeddings.shape == (1, max_len, d_model), (
        f"Expected shape (1, {max_len}, {d_model}), but got {embeddings.shape}"
    )    

def test_positional_encoding():
    d_model = 512
    max_len = 50
    pe = PositionalEncoding(d_model, max_len)
    x = torch.zeros(1, max_len, d_model)
    output = pe(x)
    assert output.shape == (1, max_len, d_model)

    # Check that all values are unique
    unique_rows = torch.unique(output, dim=0) 
    assert unique_rows.size(1) == max_len, (
        f"Expected {max_len} unique position encodings, but got "
        f"{unique_rows.size(1)}"
    )
