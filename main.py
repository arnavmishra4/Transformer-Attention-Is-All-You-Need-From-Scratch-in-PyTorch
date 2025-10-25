import torch
import torch.nn as nn
import torch.optim as optim

from model.transformer import TransformerModel
from config import *
from predict import predict
from DataLoader import get_dataloader


def main():
    # ---------------------------
    # 1. Load Data & Vocabulary
    # ---------------------------
    dataloader, vocab = get_dataloader()
    vocab_list = list(vocab.keys())   # list of tokens for model embedding init

    # ---------------------------
    # 2. Initialize Model, Loss, Optimizer
    # ---------------------------
    model = TransformerModel(
        vocab_list,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        d_ff=d_ff
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ---------------------------
    # 3. Training Loop
    # ---------------------------
    print("\n🚀 Starting training...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src_batch, tgt_in_batch, tgt_out_batch in dataloader:
            src_batch, tgt_in_batch, tgt_out_batch = (
                src_batch.to(device),
                tgt_in_batch.to(device),
                tgt_out_batch.to(device)
            )

            optimizer.zero_grad()

            # forward pass
            logits = model(src_batch, tgt_in_batch)  # [B, seq_len, vocab_size]
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out_batch.view(-1))

            # backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}]  Avg Loss: {avg_loss:.4f}")

    # ---------------------------
    # 4. Save Trained Model
    # ---------------------------
    torch.save(model.state_dict(), "checkpoints/model.pt")
    print("\n✅ Model saved at checkpoints/model.pt")

    # ---------------------------
    # 5. Run Sample Prediction
    # ---------------------------
    print("\n--- Inference ---")
    model.eval()
    sample_input = ["hey", "how"]
    generated = predict(model, sample_input, vocab, max_words=5, device=device)
    print("Input: ", " ".join(sample_input))
    print("Generated:", " ".join(generated))


if __name__ == "__main__":
    main()
