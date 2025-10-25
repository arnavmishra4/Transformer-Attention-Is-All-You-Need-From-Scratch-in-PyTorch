import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import TransformerModel
from utils import tokens_to_tensor
from config import d_model, n_head, d_ff, learning_rate, epochs


def train_model(data, src, tgt_in, tgt_out, vocab):
    # --- initialize model ---
    model = TransformerModel(data, d_model=d_model, n_head=n_head, num_layers=num_layers, d_ff=d_ff)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- convert tokens to tensors ---
    src_tensor = tokens_to_tensor(src, vocab)
    tgt_in_tensor = tokens_to_tensor(tgt_in, vocab)
    tgt_out_tensor = tokens_to_tensor(tgt_out, vocab)

    # --- training loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(src_tensor, tgt_in_tensor)  # [B, seq_len, vocab_size]
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_out_tensor.view(-1))

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/{epochs}]  Loss: {loss.item():.4f}")

    return model
