# 🧠 Transformer (Attention Is All You Need) — From Scratch in PyTorch

> *A from-scratch PyTorch implementation of the "Attention Is All You Need" Transformer architecture — my full researcher-level rebuild of the encoder-decoder attention model.*

---

## 🌍 Overview

This repository contains a complete **from-scratch implementation** of the Transformer model (Vaswani et al., 2017).  
It recreates the full **Encoder–Decoder pipeline** including attention, positional encoding, feed-forward layers, and decoding — all written manually in PyTorch without using built-in transformer modules.

This project is primarily meant for **learning, experimentation, and research**. It demonstrates how the core mechanics of the Transformer architecture actually work under the hood.

---

## 🗂️ Repository Structure

```
transformer_project/
│
├── data/
│   ├── train.json          # training samples (input-target pairs)
│   └── val.json            # optional validation data
│
├── model/
│   └── transformer.py      # full Transformer implementation
│
├── utils.py                # helper functions (load_json, vocab builder, tensor conversion)
├── train.py                # modular training script
├── predict.py              # greedy decoding / inference
├── config.py               # hyperparameters and paths
└── main.py                 # entry point for training + prediction
```

---

## ⚙️ Installation

**Requirements**

```
Python >= 3.10
torch >= 2.0
```

**Install dependencies**

```bash
pip install torch
```

---

## 📘 Dataset Format

All training data should be stored inside `/data/train.json` in the following format:

```json
[
  {
    "input": ["hey", "how", "are", "you"],
    "target": ["<bos>", "hey", "how", "are", "you", "<eos>"]
  },
  {
    "input": ["how", "are", "you"],
    "target": ["<bos>", "how", "are", "you", "<eos>"]
  }
]
```

Each sample contains:

* `"input"` → source sentence tokens (encoder input)
* `"target"` → expected output tokens (decoder target sequence)
* `<bos>` and `<eos>` tokens define sequence boundaries during generation

---

## ⚙️ Configuration

All global settings (paths, model dimensions, hyperparameters) are defined in `config.py`:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
train_path = "data/train.json"
learning_rate = 1e-4
epochs = 10
d_model = 512
n_head = 8
d_ff = 2048
num_layers = 2
```

---

## 🚀 Training

To start training the Transformer model, simply run:

```bash
python main.py
```

This will:

1. Load and preprocess the dataset
2. Build the vocabulary
3. Initialize the Transformer model
4. Train for the specified number of epochs
5. Save the trained model to `/checkpoints/model.pt`
6. Automatically run a sample inference after training

**Example output:**

```
Epoch [1/10]  Loss: 3.9231
Epoch [2/10]  Loss: 2.8410
✅ Model saved at checkpoints/model.pt

--- Inference ---
Generated sequence: hey how are you
```

---

## 🧩 Inference / Text Generation

Once trained, you can generate predictions using the `predict.py` module.
It uses greedy decoding to predict tokens one at a time.

```python
from predict import predict
from model.transformer import TransformerModel
import torch

# load vocab and model
model = TransformerModel(vocab_list)
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.to("cuda")

# generate output
generated = predict(model, ["hey", "how"], vocab, max_words=10, device="cuda")
print("Generated:", generated)
```

**Expected Output**

```
Generated: ['hey', 'how', 'are', 'you']
```

---

## 🧱 Key Components

| File | Description |
|------|-------------|
| **`model/transformer.py`** | Core Transformer implementation (Encoder, Decoder, Attention, FFN, etc.) |
| **`utils.py`** | Helper utilities for loading JSON data, creating vocabularies, and converting tokens to tensors |
| **`train.py`** | Training script with loss, optimizer, and checkpoint saving |
| **`predict.py`** | Greedy decoding for sequence generation |
| **`config.py`** | Hyperparameters, file paths, and device configuration |
| **`main.py`** | Entry point — runs training and inference |

---

## 💾 Saving & Loading Models

The trained model is automatically saved in:

```
checkpoints/model.pt
```

You can later reload it using:

```python
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.eval()
```

---

## 🧠 Purpose

This project is built to help understand the **internal mechanics of Transformer models** —
especially how multi-head attention, positional encoding, and residual connections work when written manually in PyTorch.

It's not optimized for large-scale NLP training — it's meant for **researchers, students, and engineers** who want to dig into the architecture that started modern generative AI.

---

## 🧔 Author

**Arnav Mishra**  
AI Researcher | Deep Learning Enthusiast

This project was built line-by-line to fully understand the Transformer model from scratch — an educational, research-oriented implementation of *Attention Is All You Need* in PyTorch.

---

## ⭐ Acknowledgements

Inspired by the original paper:

> Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS 2017.

Special thanks to the open-source PyTorch community for providing reference code and documentation.

---

### ⭐ If you found this helpful, consider giving it a star on GitHub!
