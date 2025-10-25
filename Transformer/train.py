import torch
import torch.nn as nn
import torch.optim as optim

from model.transformer import TransformerModel
from utils import load_json, build_vocab, tokens_to_tensor
from config import *


def train():
    train_data = load_json(train_path)
    vocab = build_vocab(train_data)

    sample = train_data[0]
    src = sample["input"]
    tgt_in = sample["target"][:-1]    # all except last
    tgt_out = sample["target"][1:]    # all except first

    src_tensor = tokens_to_tensor(src, vocab).to(device)
    tgt_in_tensor = tokens_to_tensor(tgt_in, vocab).to(device)
    tgt_out_tensor = tokens_to_tensor(tgt_out, vocab).to(device)

    model = TransformerModel(vocab, d_model=d_model, n_head=n_head,
                             num_layers=num_layers, d_ff=d_ff).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(src_tensor, tgt_in_tensor)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_out_tensor.view(-1))

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/{epochs}]  Loss: {loss.item():.4f}")

    # -------------------------
    # 💾 Save trained model
    # -------------------------
    torch.save(model.state_dict(), "checkpoints/model.pt")
    print("\n✅ Model saved at checkpoints/model.pt")

    return model, vocab
