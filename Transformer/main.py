import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import TransformerModel
from config import *
from predict import predict
from utils import load_json, build_vocab, tokens_to_tensor
from config import train_path

train_data = load_json(train_path)
vocab = build_vocab(train_data)

sample = train_data[0]
src = sample["input"]
tgt_in = sample["target"][:-1]    # all except last
tgt_out = sample["target"][1:]    # all except first


# ---------------------------
# 🚀 1. Instantiate model, loss, optimizer
# ---------------------------
model = TransformerModel(Data, d_model=d_model, n_head=n_head, num_layers=num_layers, d_ff=d_ff)
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------------------
# 🧩 2. Convert to tensors
# ---------------------------
src_tensor = tokens_to_tensor(src, vocab).to(device)
tgt_in_tensor = tokens_to_tensor(tgt_in, vocab).to(device)
tgt_out_tensor = tokens_to_tensor(tgt_out, vocab).to(device)

# ---------------------------
# 🧠 3. Training loop
# ---------------------------
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(src_tensor, tgt_in_tensor)
    loss = criterion(logits.view(-1, logits.size(-1)), tgt_out_tensor.view(-1))

    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch}/{epochs}]  Loss: {loss.item():.4f}")

# ---------------------------
# ✨ 4. Prediction after training
# ---------------------------
print("\n--- Inference ---")
prediction = predict(model, ['hey', 'how'], vocab, max_words=5)
print("Generated:", prediction)
