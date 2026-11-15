








from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np
import torch
from .graph import induced_graph_mask


@dataclass(frozen=True)
class GeneChunk:

    selected: torch.Tensor
    core: torch.Tensor
    core_positions: torch.Tensor


def build_gene_chunks(neighbors: torch.Tensor, *, core_size: int = 96,
                      max_context: int = 256, order=None) -> list[GeneChunk]:





    if neighbors.ndim != 2 or neighbors.dtype != torch.long:
        raise ValueError('neighbors must be integer [G,K]')
    genes = neighbors.shape[0]
    if genes < 1 or core_size < 1 or max_context < core_size:
        raise ValueError('Require genes>=1 and 1<=core_size<=max_context')
    if order is None:
        order = np.arange(genes, dtype=np.int64)
    else:
        order = np.asarray(order, dtype=np.int64)
        if len(order) != genes or not np.array_equal(np.sort(order), np.arange(genes)):
            raise ValueError('order must be a permutation of all gene indices')
    neighbor_list = neighbors.cpu().tolist()
    if any(j < 0 or j >= genes for row in neighbor_list for j in row):
        raise ValueError('Graph contains a neighbor outside gene vocabulary')
    result = []
    for i in range(0, genes, core_size):
        core = list(map(int, order[i:i+core_size]))
        selected = set(core)
        queue = deque(core)
        
        while queue and len(selected) < max_context:
            node = queue.popleft()
            for neighbor in neighbor_list[node]:
                if neighbor not in selected:
                    selected.add(neighbor)
                    queue.append(neighbor)
                    if len(selected) == max_context:
                        break
        indexed = sorted(selected)
        positions = {gene: n for n, gene in enumerate(indexed)}
        result.append(GeneChunk(
            selected=torch.tensor(indexed, dtype=torch.long),
            core=torch.tensor(core, dtype=torch.long),
            core_positions=torch.tensor([positions[x] for x in core], dtype=torch.long),
        ))
    return result


@torch.inference_mode()
def generate_chunked(model, control: torch.Tensor, perturbations: torch.Tensor,
                     graph_neighbors: torch.Tensor, control_std: torch.Tensor,
                     *, steps: int = 20, tau: float = 0.1, core_size: int = 96,
                     max_context: int = 256, integrator: str = 'euler',
                     ablate=(), clamp_output: bool = False, generator=None,
                     chunks: list[GeneChunk] | None = None) -> torch.Tensor:





    if control.ndim != 2 or control.shape[1] != len(graph_neighbors):
        raise ValueError('Control expression must be [B, all_checkpoint_genes]')
    if control_std.ndim != 1 or len(control_std) != control.shape[1]:
        raise ValueError('Control std must have a value for each gene')
    if chunks is None:
        chunks = build_gene_chunks(graph_neighbors, core_size=core_size,
                                   max_context=max_context)
    out = torch.empty_like(control)
    coverage = torch.zeros(control.shape[1], device=control.device, dtype=torch.int32)
    for chunk in chunks:
        idx = chunk.selected.to(control.device)
        core = chunk.core.to(control.device)
        core_positions = chunk.core_positions.to(control.device)
        mask = induced_graph_mask(graph_neighbors, chunk.selected.cpu(),
                                  disable_mask='gene_mask' in ablate).to(control.device)
        estimate = model.generate(
            control.index_select(1, idx), perturbations, idx, mask, control_std,
            steps=steps, tau=tau, ablate=ablate, integrator=integrator,
            clamp_output=clamp_output, generator=generator)
        out[:, core] = estimate[:, core_positions]
        coverage.index_add_(0, core, torch.ones_like(core, dtype=torch.int32))
    if not torch.all(coverage == 1):
        raise RuntimeError('Chunk plan must cover each gene exactly once')
    return out
