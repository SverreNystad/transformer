import torch

from src.embedder import END_TOKEN, START_TOKEN, Embedder, make_causal_mask
from src.network import Transformer


@torch.no_grad()
def inspect_predictions(
    model: Transformer,
    embedder: Embedder,
    device: str = "cpu",
):
    model.eval()
    model.to(device)
    embedder.to(device)

    end_id = embedder.vocab_table[END_TOKEN]

    generated_ids = [START_TOKEN]
    src_emb = embedder(START_TOKEN).to(device)  # (1, src_len, d_model)
    tgt_emb = src_emb

    for _ in range(15):
        # embed current prefix
        sentence = "".join([embedder.get_token(i) for i in generated_ids])
        src_emb = embedder(sentence).to(device)  # (1, seq_len, d_model)
        tgt_emb = src_emb

        # causal mask
        cur_len = tgt_emb.size(1)
        mask = make_causal_mask(cur_len, device).unsqueeze(0).unsqueeze(1)  # (1,1,cur_len,cur_len)

        # forward & pick next
        logits = model(src_emb, tgt_emb, mask)  # (1, cur_len, vocab_size)
        next_id = int(logits[0, -1].argmax().item())  # greedy

        generated_ids.append(next_id)
        if next_id == end_id:
            break

    generated_tokens = [embedder.get_token(i) for i in generated_ids]
    print("Generated :", " ".join(generated_tokens))
