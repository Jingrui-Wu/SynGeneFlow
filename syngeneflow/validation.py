
from __future__ import annotations
import torch


def validate_flow_batch(control, target, perturbations, indices, adjacency, control_std):
    if control.ndim != 2 or target.shape != control.shape:
        raise ValueError('Source and target should both be [B,G]')
    b, genes = control.shape
    if b < 2:
        raise ValueError('Need at least two unpaired cell samples per condition')
    if perturbations.ndim != 2 or perturbations.shape[0] != b:
        raise ValueError('Perturbations must be padded integer IDs [B,P]')
    if indices.ndim != 1 or len(indices) != genes:
        raise ValueError('Gene subset indices must match expression columns')
    if adjacency.shape != (genes, genes) or adjacency.dtype != torch.bool:
        raise ValueError('Gene graph must be boolean [G,G]')
    if not torch.all(adjacency.diagonal()):
        raise ValueError('Coexpression attention graph must contain self edges')
    if not torch.equal(adjacency, adjacency.T):
        raise ValueError('Coexpression graph must be symmetric')
    if (control_std < 0).any() or not torch.isfinite(control_std).all():
        raise ValueError('Invalid training control standard deviations')
    if not torch.isfinite(control).all() or not torch.isfinite(target).all():
        raise ValueError('Unpaired expression contains NaN or Inf')


def validate_condition_homogeneity(perturbation_ids):
    if not torch.equal(perturbation_ids, perturbation_ids[:1].expand_as(perturbation_ids)):
        raise ValueError('Distributional alignment requires same-condition mini-batches')
