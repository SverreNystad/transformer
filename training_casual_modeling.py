import os

import torch
from dotenv import load_dotenv
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import wandb
from src.data import get_dataloader
from src.embedder import Embedder, make_causal_mask
from src.network import Transformer

load_dotenv()


def train(
    sentences: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str = "cpu",
):
    # Hyperparameters (must match your Embedder and Transformer)
    d_model = 4
    n_heads = 2
    d_ff = 8
    num_layers = 2
    max_seq_len = 6
    wandb.init(
        project="causal-transformer",
        # mode="disabled",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "max_seq_len": max_seq_len,
        },
    )

    # Instantiate embedder & model
    embedder = Embedder(embedding_dimension=d_model, max_seq_len=max_seq_len).to(device)
    vocab_size = embedder.vocab_size

    model = Transformer(
        output_classes=vocab_size,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
    ).to(device)

    wandb.watch(model, log="all")

    # Loss ignores padding token id = 0
    criterion = CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=lr)

    dataloader = get_dataloader(sentences, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            # batch is a list of strings (length=batch_size)
            sentence = batch[0]  # since batch_size=1

            # Embed source & target
            src_emb = embedder(sentence).to(device)  # (1, seq_len, d_model)
            tgt_emb = src_emb  # causal LM uses decoder only

            # Build masks
            seq_len = src_emb.size(1)
            tgt_mask = make_causal_mask(seq_len, device)

            # Forward pass
            logits = model(src_emb, tgt_emb, tgt_mask)  # (1, seq_len, vocab_size)
            logits = logits.squeeze(0)  # → (seq_len, vocab_size)

            # Build gold token IDs (same logic as Embedder)
            doc = embedder.nlp(sentence)
            token_ids = [token.idx for token in doc]
            if len(token_ids) < max_seq_len:
                token_ids += [0] * (max_seq_len - len(token_ids))
            gold = torch.tensor(token_ids, dtype=torch.long, device=device)

            # Shift for next-token loss: predict at t from inputs <t
            pred = logits[:-1, :]  # positions 0…T-2 predict 1…T-1
            target = gold[1:]  # positions 1…T-1
            loss = criterion(pred, target)

            # Backprop & optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:02d} — Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": avg_loss})

        if epoch % 100 == 0:
            checkpoint_path = f"models/model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            artifact = wandb.Artifact("transformer-model", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            print(f"Model checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    # Wandb logging
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)

    sentences = [
        "Dette er en testsetning",
        "Maskinlæring er gøy",
        "Hvordan går det i dag",
        "Jeg liker transformer arkitektur",
        "Cogito for LIFE!",
    ]
    train(sentences, epochs=2000, batch_size=10, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu")
