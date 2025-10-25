import torch 
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2056, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)
