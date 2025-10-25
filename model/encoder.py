import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.embedding import Embedding, PositionalEncoding


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, n_head=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.mha(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, data, d_model=512, n_head=8, num_layers=6, d_ff=2048):
        super().__init__()
        self.embedding = Embedding(data)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model=d_model, n_head=n_head, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embedding(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
