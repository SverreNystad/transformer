import os

import torch
from dotenv import load_dotenv
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import wandb
from inference import inspect_predictions
from src.data import JokesDataset
from src.embedder import PADDING_TOKEN, Embedder, create_vocab_mapping, make_causal_mask
from src.network import Transformer

load_dotenv()


class Hyperparameters:
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        lr: float,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len


def train(
    hyper_params: Hyperparameters,
    dataset: Dataset,
    device: str = "cpu",
    shall_save: bool = False,
):
    # Hyperparameters
    epochs = hyper_params.epochs
    batch_size = hyper_params.batch_size
    lr = hyper_params.lr
    d_model = hyper_params.d_model
    n_heads = hyper_params.n_heads
    d_ff = hyper_params.d_ff
    num_layers = hyper_params.num_layers
    max_seq_len = hyper_params.max_seq_len

    wandb.init(
        project="causal-transformer",
        mode="disabled",
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    sentences = []
    for batch in dataloader:
        sentences.extend(batch)
    language = "norwegian"
    vocab: dict[str, int] = create_vocab_mapping(sentences, language)
    # save vocab to file
    with open("vocab.txt", "w") as f:
        for word, idx in vocab.items():
            f.write(f"{word}\t{idx}\n")

    # Instantiate embedder & model
    embedder = Embedder(vocab_table=vocab, embedding_dimension=d_model, max_seq_len=max_seq_len).to(device)
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

    # Loss ignores padding token
    padding_id = embedder.vocab_table[PADDING_TOKEN]
    criterion = CrossEntropyLoss(ignore_index=padding_id)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            # batch is a list of strings (length=batch_size)
            sentence = batch[0]

            # Embed source & target
            src_emb = embedder(sentence).to(device)  # (1, seq_len, d_model)
            tgt_emb = src_emb

            # Build masks
            seq_len = src_emb.size(1)
            tgt_mask = make_causal_mask(seq_len, device)

            # Forward pass
            logits = model(src_emb, tgt_emb, tgt_mask)  # (1, seq_len, vocab_size)
            logits = logits.squeeze(0)  # (seq_len, vocab_size)

            # Build gold token IDs (same logic as Embedder)
            token_ids = embedder.get_token_ids(sentence, language=language)
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

        # Inference
        inspect_predictions(
            model=model,
            embedder=embedder,
            device=device,
        )

        if epoch % 100 == 0 and shall_save:
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

    hyper_params = Hyperparameters(
        epochs=2000,
        batch_size=1,
        lr=0.001,
        d_model=512,
        n_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=200,
    )

    jokes_dataset = JokesDataset()

    train(
        hyper_params=hyper_params,
        dataset=jokes_dataset,
        device="cuda" if torch.cuda.is_available() else "cpu",
        shall_save=False,
    )
