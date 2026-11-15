
from __future__ import annotations
import torch
from torch import nn
from .nn.gene_attention import GraphGeneEncoder
from .nn.velocity_attention import VelocityBlock
from .nn.time_embedding import TimeEmbedding
from .transport.integrators import integrate

class SynGeneFlow(nn.Module):
    def __init__(self, n_genes, n_perturbations, control_id, *,
                 perturb_to_gene=None, hidden=128, heads=4,
                 gene_layers=2, velocity_layers=4, edge_hidden=32):
        super().__init__()
        if hidden % heads:
            raise ValueError('hidden must be divisible by heads')
        if not 0 <= control_id < n_perturbations:
            raise ValueError('control_id must be an integer within the perturbation vocabulary')
        self.n_genes = n_genes
        self.n_perturbations = n_perturbations
        self.control_id = control_id
        self.gene_embedding = nn.Embedding(n_genes, hidden)
        self.perturbation_embedding = nn.Embedding(n_perturbations, hidden,
                                                   padding_idx=control_id)
        if perturb_to_gene is None:
            perturb_to_gene = [-1] * n_perturbations
        if len(perturb_to_gene) != n_perturbations:
            raise ValueError('perturb_to_gene must have n_perturbations entries')
        self.register_buffer('perturb_to_gene', torch.as_tensor(perturb_to_gene, dtype=torch.long))
        self.expression_encoder = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(),
                                                nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.gene_encoder = GraphGeneEncoder(hidden, heads, gene_layers, edge_hidden)
        
        self.single_effect_decoder = nn.Sequential(nn.Linear(3 * hidden, 2 * hidden),
                                                    nn.SiLU(), nn.Linear(2 * hidden, hidden),
                                                    nn.SiLU(), nn.Linear(hidden, 1))
        self.time_embedder = TimeEmbedding(hidden)
        self.velocity_blocks = nn.ModuleList([VelocityBlock(hidden, heads)
                                              for _ in range(velocity_layers)])
        self.velocity_head = nn.Sequential(nn.LayerNorm(2 * hidden),
                                           nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                           nn.Linear(hidden, 1))
        self.gene_norm = nn.LayerNorm(hidden)

    def perturbation_tokens(self, ids):





        mapped = self.perturb_to_gene[ids]
        embedded_gene = self.gene_embedding(mapped.clamp(min=0))
        embedded_drug = self.perturbation_embedding(ids)
        tokens = torch.where((mapped >= 0).unsqueeze(-1), embedded_gene, embedded_drug)
        return tokens * (ids != self.control_id).unsqueeze(-1)

    def perturbation_context(self, ids):
        if ids.ndim == 1:
            ids = ids[:, None]
        return self.perturbation_tokens(ids).sum(dim=1)

    def additive_prior(self, control, pert_ids, gene_indices):

        if pert_ids.ndim == 1:
            pert_ids = pert_ids[:, None]
        B, N = control.shape
        genes = self.gene_norm(self.gene_embedding(gene_indices))[None].expand(B, -1, -1)
        expression = self.expression_encoder(control.unsqueeze(-1))
        c = expression + genes
        mean_context = c.mean(dim=1, keepdim=True).expand_as(c)
        components = self.perturbation_tokens(pert_ids)
        shifts = control.new_zeros(B, N)
        for j in range(pert_ids.shape[1]):
            mask = (pert_ids[:,j] != self.control_id).to(control.dtype).unsqueeze(-1)
            p = components[:,j,None,:].expand_as(c)
            delta = self.single_effect_decoder(torch.cat([c, mean_context, p], dim=-1)).squeeze(-1)
            shifts = shifts + mask * delta
        return control + shifts

    def velocity(self, xt, t, control, pert_ids, gene_indices, graph_mask,
                 *, use_deformation=True, use_differential=True):

        B, N = xt.shape
        if t.ndim == 0:
            t = t.expand(B)
        if t.ndim == 2:
            t = t.squeeze(-1)
        p = self.perturbation_context(pert_ids)
        h = self.gene_embedding(gene_indices)[None].expand(B, -1, -1)
        z = self.gene_encoder(h, p, graph_mask, use_deformation=use_deformation)
        hc = self.expression_encoder(control.unsqueeze(-1)) + z
        hv = self.expression_encoder(xt.unsqueeze(-1)) + z
        t_emb = self.time_embedder(t)
        for block in self.velocity_blocks:
            hv = block(hv, hc, p, t_emb, use_differential=use_differential)
        h = torch.cat([hv, p[:,None,:].expand_as(hv)], dim=-1)
        return self.velocity_head(h).squeeze(-1)

    forward = velocity

    @torch.no_grad()
    def generate(self, control, pert_ids, gene_indices, graph_mask, control_std,
                 *, steps=20, tau=0.1, ablate=(), generator=None,
                 integrator="euler", clamp_output=False, return_trajectory=False):





        ablate = set(ablate)
        if steps < 1:
            raise ValueError('steps must be >= 1')
        if 'additive_prior' in ablate:
            x = torch.randn(control.shape, device=control.device, dtype=control.dtype,
                            generator=generator)
        else:
            anchor = self.additive_prior(control, pert_ids, gene_indices)
            noise = torch.randn(control.shape, device=control.device, dtype=control.dtype,
                                generator=generator) if tau else torch.zeros_like(control)
            x = anchor + float(tau) * noise * control_std[gene_indices].unsqueeze(0)
        if 'residual_flow' in ablate:
            return x.clamp_min(0.) if clamp_output else x
        def velocity_field(state, t):
            return self.velocity(state, t, control, pert_ids, gene_indices, graph_mask,
                                 use_deformation="deformation" not in ablate,
                                 use_differential="differential_attention" not in ablate)
        x = integrate(x, velocity_field, steps=steps, method=integrator,
                      return_trajectory=return_trajectory)
        return x.clamp_min(0.) if clamp_output else x

