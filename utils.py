# utils.py
import json
import torch

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def build_vocab(data, pad="<pad>", bos="<bos>", eos="<eos>"):
    words = set()
    for sample in data:
        words.update(sample["input"] + sample["target"])  # assume dict format
    vocab = {pad: 0, bos: 1, eos: 2}
    for i, w in enumerate(sorted(words), start=3):
        vocab[w] = i
    return vocab

def tokens_to_tensor(tokens, vocab):
    ids = [vocab[w] for w in tokens]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
