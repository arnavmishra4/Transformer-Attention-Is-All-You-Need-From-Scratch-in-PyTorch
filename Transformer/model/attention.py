import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attn_dim = d_model // n_head
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K=None, V=None, mask=None):
        if K is None: K = Q
        if V is None: V = Q

        B, L, D = Q.shape
        attn_dim = D // self.n_head
        Q = self.W_Q(Q).view(B, L, self.n_head, attn_dim).transpose(1, 2)
        K = self.W_K(K).view(B, K.size(1), self.n_head, attn_dim).transpose(1, 2)
        V = self.W_V(V).view(B, V.size(1), self.n_head, attn_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (attn_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = self.softmax(attn_scores)
        attn_out = torch.matmul(attn_probs, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.W_O(attn_out)

        return out
