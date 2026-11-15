
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

class DifferentialAttention(nn.Module):





    def __init__(self, hidden, heads):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.d = hidden // heads
        self.q1 = nn.Linear(hidden, hidden)
        self.q2 = nn.Linear(hidden, hidden)
        self.k1 = nn.Linear(hidden, hidden)
        self.k2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.lam = nn.Parameter(torch.zeros(heads))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, context):
        B, N, H = x.shape
        M = context.shape[1]
        def as_heads(t, seq):
            return t.reshape(B, seq, self.heads, self.d).transpose(1, 2)
        q1, q2 = as_heads(self.q1(x), N), as_heads(self.q2(x), N)
        k1, k2 = as_heads(self.k1(context), M), as_heads(self.k2(context), M)
        v = as_heads(self.v(context), M)
        a1 = F.softmax(((q1 @ k1.transpose(-1,-2)) / math.sqrt(self.d)).float(), -1)
        a2 = F.softmax(((q2 @ k2.transpose(-1,-2)) / math.sqrt(self.d)).float(), -1)
        diff = a1 - self.lam.sigmoid()[None, :, None, None] * a2
        z = (diff.to(v.dtype) @ v).transpose(1, 2).reshape(B, N, H)
        return self.out(self.norm(z))


class TimeConditionedNorm(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mod = nn.Linear(hidden, 2 * hidden)

    def forward(self, x, time_emb):
        shift, scale = self.mod(time_emb).chunk(2, dim=-1)
        return self.norm(x) * (1. + scale.unsqueeze(1)) + shift.unsqueeze(1)


class VelocityBlock(nn.Module):

    def __init__(self, hidden, heads):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                       nn.Linear(hidden, hidden))
        self.n1 = TimeConditionedNorm(hidden)
        self.n2 = TimeConditionedNorm(hidden)
        self.n3 = TimeConditionedNorm(hidden)
        self.standard_self = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.standard_cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.differential_self = DifferentialAttention(hidden, heads)
        self.differential_cross = DifferentialAttention(hidden, heads)
        self.ffn = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                                 nn.Linear(4 * hidden, hidden))

    def forward(self, x, control, p, t, use_differential=True):
        x = x + self.condition(torch.cat([x, p[:,None,:].expand_as(x)], dim=-1))
        y = self.n1(x, t)
        if use_differential:
            x = x + self.differential_self(y, y)
        else:
            x = x + self.standard_self(y, y, y, need_weights=False)[0]
        y = self.n2(x, t)
        if use_differential:
            x = x + self.differential_cross(y, control)
        else:
            x = x + self.standard_cross(y, control, control, need_weights=False)[0]
        x = x + self.ffn(self.n3(x, t))
        return x


