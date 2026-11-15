




from __future__ import annotations

import numpy as np
import torch


def _to_dense_float(X):
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def topk_coexpression_neighbors(X, k=30, chunk_size=128):





    X = _to_dense_float(X)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError('Requires >= 2 training cells and 2D expression matrix')
    n, g = X.shape
    if not 0 <= k < g:
        k = min(max(k, 0), g - 1)
    if k == 0:
        return torch.empty((g, 0), dtype=torch.long)
    X = np.nan_to_num(X, copy=True)
    X -= X.mean(axis=0, keepdims=True)
    norm = np.sqrt((X * X).sum(axis=0))
    X /= np.maximum(norm[None], 1e-8)
    neighbors = np.empty((g, k), dtype=np.int64)
    for start in range(0, g, chunk_size):
        end = min(start + chunk_size, g)
        corr = np.abs(X[:, start:end].T @ X)
        corr[np.arange(end - start), np.arange(start, end)] = -np.inf
        top = np.argpartition(-corr, k - 1, axis=1)[:, :k]
        strength = np.take_along_axis(corr, top, axis=1)
        neighbors[start:end] = np.take_along_axis(top, np.argsort(-strength, axis=1), axis=1)
    return torch.from_numpy(neighbors)


def induced_graph_mask(neighbors, indices, disable_mask=False):





    if indices.dim() != 1:
        raise ValueError('indices must be a 1D expression-column index vector')
    n = indices.numel()
    if disable_mask:
        return torch.ones(n, n, device=indices.device, dtype=torch.bool)
    selected = indices.tolist()
    pos = {int(g): i for i, g in enumerate(selected)}
    mask = torch.eye(n, device=indices.device, dtype=torch.bool)
    
    for i, g in enumerate(selected):
        for neighbor in neighbors[int(g)].tolist():
            j = pos.get(int(neighbor))
            if j is not None:
                mask[i, j] = True
                mask[j, i] = True
    return mask


def sample_connected_subset(neighbors, n_selected, generator=None):

    g = neighbors.shape[0]
    n_selected = min(n_selected, g)
    if n_selected <= 0:
        raise ValueError('n_selected must be positive')
    ordering = torch.randperm(g, generator=generator)
    selected, used = [], set()
    for seed in ordering.tolist():
        for candidate in [seed, *neighbors[seed].tolist()]:
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
                if len(selected) >= n_selected:
                    return torch.tensor(sorted(selected), dtype=torch.long)
    return torch.tensor(sorted(selected), dtype=torch.long)
