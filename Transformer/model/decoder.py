import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.embedding import Embedding, PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, data, d_model=512, n_head=8, d_ff=2048):
        super().__init__()
        self.MHA_1 = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.MHA_2 = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.pos = PositionalEncoding(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        self.embedding = Embedding(data)
        self.dropout = nn.Dropout()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tokens, encoder_out, mask=None):
        # Embed + positional encoding
        x = self.embedding(tokens)
        x = self.pos(x)

        # Self-attention within decoder
        attn_out_1 = self.MHA_1(Q=x, K=x, V=x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out_1))

        # Cross-attention with encoder output
        attn_out_2 = self.MHA_2(Q=x, K=encoder_out, V=encoder_out)
        x = self.norm2(x + self.dropout(attn_out_2))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x
