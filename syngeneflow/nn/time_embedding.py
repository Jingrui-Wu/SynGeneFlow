
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.SiLU(),
                                 nn.Linear(hidden * 4, hidden))

    def forward(self, t):
        if t.ndim == 0:
            t = t[None]
        dim = self.hidden // 2
        freq = torch.exp(-math.log(10000.) * torch.arange(dim, device=t.device,
                                                         dtype=t.dtype) / max(dim - 1, 1))
        phase = t[:, None] * freq[None] * 1000.
        emb = torch.cat([phase.cos(), phase.sin()], dim=-1)
        if emb.shape[-1] < self.hidden:
            emb = F.pad(emb, (0, self.hidden - emb.shape[-1]))
        return self.mlp(emb)


