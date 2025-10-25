import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.embedding import Embedding
from config import d_model, n_head, d_ff, batch_size

class TransformerModel(nn.Module):
    def __init__(self, data, d_model=d_model, n_head=n_head, num_layers=batch_size, d_ff=d_ff):
        super().__init__()
        vocab_size = len(data)
        
        # Shared embedding across encoder + decoder + linear
        shared_embedding = Embedding(data)

        self.encoder = Encoder(
            data=data,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            d_ff=d_ff
        )

        self.decoder = Decoder(
            data=data,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff
        )

        self.linear = nn.Linear(d_model, vocab_size)
        # Tie output weights to embedding weights (common trick)
        self.linear.weight = shared_embedding.embedding.weight

    def forward(self, src, tgt, mask=None, return_probs=False):
        # ---- Encode ----
        encoder_out = self.encoder(src)  # [B, seq_len, d_model]

        # ---- Decode ----
        decoder_out = self.decoder(tgt, encoder_out, mask=mask)  # [B, seq_len, d_model]

        # ---- Project to vocab logits ----
        logits = self.linear(decoder_out)  # [B, seq_len, vocab_size]

        # ---- (Optional) Softmax for probabilities ----
        if return_probs:
            return nn.functional.softmax(logits, dim=-1)

        return logits
