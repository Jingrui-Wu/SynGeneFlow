
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .splits import active_components


@dataclass(frozen=True)
class ResponseSignature:
    condition: str
    control_mean: np.ndarray
    additive_mean: np.ndarray | None
    observed_mean: np.ndarray
    residual_mean: np.ndarray | None

    def has_observed_single_reference(self):
        return self.additive_mean is not None


def build_signature(condition, control_cells, treatment_cells, observed_single_means):
    c = np.asarray(control_cells).mean(0)
    y = np.asarray(treatment_cells).mean(0)
    parts = active_components(condition)
    if all(gene in observed_single_means for gene in parts):
        add = c + sum((np.asarray(observed_single_means[g]) - c for g in parts),
                      np.zeros_like(c))
        residual = y - add
    else:
        add, residual = None, None
    return ResponseSignature(condition, c, add, y, residual)


def rank_nonadditive_genes(signature: ResponseSignature, gene_names, k=50):
    if signature.residual_mean is None:
        return []
    residual = signature.residual_mean
    ids = np.argsort(-np.abs(residual))[:min(k,len(residual))]
    return [(str(gene_names[i]), float(residual[i])) for i in ids]


def signature_table(signatures, gene_names, k=50):
    return [{'condition': signature.condition, 'gene': gene, 'residual_delta': value}
            for signature in signatures
            for gene, value in rank_nonadditive_genes(signature, gene_names, k=k)]
