# config.py

# -------------------------------
# 🔧 Model Hyperparameters
# -------------------------------

d_model = 512           # Embedding size
n_head = 8              # Number of attention heads
d_ff = 2048             # Feedforward dimension
num_layers = 6          # Encoder/decoder layers
dropout = 0.1           # Dropout rate

# -------------------------------
# 🧠 Training Hyperparameters
# -------------------------------

learning_rate = 1e-4
epochs = 10
batch_size = 2
max_len = 5000          # For positional encoding
pad_token = "<pad>"
bos_token = "<bos>"
eos_token = "<eos>"

# -------------------------------
# 📦 Data Settings
# -------------------------------

train_path = "data/train.json"
val_path   = "data/val.json"
test_path  = "data/test.json"

# You can update these later with your dataset directories
vocab_file = "data/vocab.json"

# -------------------------------
# 🚀 Generation Settings
# -------------------------------

max_gen_tokens = 20     # Max tokens to generate during inference
temperature = 1.0        # Optional, for sampling instead of greedy decoding

# -------------------------------
# ⚙️ Device Setup
# -------------------------------

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
