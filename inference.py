import torch

from src.embedder import (
    END_TOKEN,
    PADDING_TOKEN,
    START_TOKEN,
    Embedder,
    make_causal_mask,
)
from src.network import Transformer


@torch.no_grad()
def inspect_predictions(model: Transformer, embedder: Embedder, device="cpu") -> str:
    model.eval().to(device)
    pad_id = embedder.vocab_table[PADDING_TOKEN]
    start_id = embedder.vocab_table[START_TOKEN]
    end_id = embedder.vocab_table[END_TOKEN]
    max_len = embedder.max_seq_len

    generated = [start_id]
    for _ in range(max_len - 1):
        # build & pad/truncate your prefix to exactly max_len
        seq = generated[-max_len:]
        pad_amt = max_len - len(seq)
        ids = [pad_id] * pad_amt + seq
        token_ids = torch.tensor([ids], device=device, dtype=torch.long)

        # Get embeddings & mask
        emb = embedder.embed_ids(token_ids)  # [batch, seq_len, d_model]
        cur_len = len(seq)
        mask = make_causal_mask(cur_len, device).unsqueeze(0).unsqueeze(1)

        # forward pass
        logits = model(emb[:, :cur_len], emb[:, :cur_len], mask)

        # Get the last token's logits
        # Index into last token, all vocab
        vocab_logits = logits[0, 0, -1, :]  # shape (vocab_size,)
        next_id = int(vocab_logits.argmax().item())
        generated.append(next_id)

        if next_id == end_id:
            break

    # finally decode once
    generated_tokens = [embedder.get_token(i) for i in generated]
    output = " ".join(generated_tokens)
    print("Generated:", output)
    return output
