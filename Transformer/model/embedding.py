import torch
import torch.nn as nn  

class Embedding(nn.Module):
    def __init__(self, data, d_model=512, pad_token="<pad>"):
        super().__init__()
        if pad_token not in data:
            data = [pad_token] + list(data)

        self.pad_token = pad_token
        self.vocab = {word: i for i, word in enumerate(data)}
        self.pad_token_id = 0
        self.embedding = nn.Embedding(len(self.vocab), d_model, padding_idx=0)

    def forward(self, x):
        # Handles both strings (token lists) and tensor IDs
        if isinstance(x, list) and isinstance(x[0], str):
            word_ids = torch.tensor([self.vocab[w] for w in x], dtype=torch.long)
        else:
            word_ids = x
        return self.embedding(word_ids)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
