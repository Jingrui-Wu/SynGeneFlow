
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

class ConditionalGeneAttention(nn.Module):





    def __init__(self, hidden, heads, edge_hidden=32):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.d = hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.edge_mlp = nn.Sequential(nn.Linear(hidden * 3, edge_hidden),
                                      nn.SiLU(), nn.Linear(edge_hidden, 1))

    def forward(self, genes, perturbation, mask, use_deformation=True):
        B, N, H = genes.shape
        q = self.q(genes).reshape(B, N, self.heads, self.d).transpose(1, 2)
        k = self.k(genes).reshape(B, N, self.heads, self.d).transpose(1, 2)
        v = self.v(genes).reshape(B, N, self.heads, self.d).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) * (self.d ** -0.5)
        if use_deformation:
            rows, cols = torch.where(mask)
            e = perturbation[:, None, :].expand(-1, rows.numel(), -1)
            inputs = torch.cat([genes[:, rows], genes[:, cols], e], dim=-1)
            deform = self.edge_mlp(inputs).squeeze(-1)
            flat = rows * N + cols
            bias = logits.new_zeros(B, N * N).scatter(1, flat[None].expand(B, -1), deform)
            logits = logits + bias.reshape(B, 1, N, N)
        logits = logits.masked_fill(~mask[None, None], -1e4)
        weights = F.softmax(logits.float(), dim=-1).to(v.dtype)
        out = weights @ v
        return self.proj(out.transpose(1, 2).reshape(B, N, H))


class GraphGeneEncoder(nn.Module):
    def __init__(self, hidden, heads, depth=2, edge_hidden=32):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'norm1': nn.LayerNorm(hidden),
            'attn': ConditionalGeneAttention(hidden, heads, edge_hidden),
            'norm2': nn.LayerNorm(hidden),
            'ff': nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                                nn.Linear(4 * hidden, hidden))
        }) for _ in range(depth)])

    def forward(self, gene_embed, perturb_emb, graph_mask, use_deformation=True):
        x = gene_embed
        for block in self.blocks:
            x = x + block['attn'](block['norm1'](x), perturb_emb, graph_mask,
                                   use_deformation=use_deformation)
            x = x + block['ff'](block['norm2'](x))
        return x


