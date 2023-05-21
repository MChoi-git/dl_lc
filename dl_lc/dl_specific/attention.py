import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, n_heads, model_dim):
        super().__init__()
        self.qkv_proj = nn.Linear(model_dim, model_dim * 3)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.n_heads = n_heads
        self.model_dim = model_dim


    def forward(self, inputs):
        B, S, H = inputs.shape
        q, k, v = torch.chunk(self.qkv_proj(inputs), 3)
        q = q.reshape(B, self.n_heads, S, -1)
        k = k.reshape(B, self.n_heads, S, -1)
        v = v.reshape(B, self.n_heads, S, -1)

        qk = torch.einsum("...sh,...Sh->...sS", q, k)
        attn_map = torch.softmax(qk * H ** -0.5, dim=-1)

        attention = torch.einsum("...sS,...Sh->...sh", attn_map, v).reshape(B, S, H)
        out = self.out_proj(attention)
        return out
