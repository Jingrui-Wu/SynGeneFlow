
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ExpressionDiagnostics:
    n_cells: int
    n_genes: int
    minimum: float
    maximum: float
    mean: float
    zero_fraction: float
    negative_fraction: float
    nonfinite_count: int

    def as_dict(self):
        return self.__dict__.copy()


def describe_expression(expression, *, chunk_size=4096):

    n_cells, n_genes = expression.shape
    if not n_cells or not n_genes:
        raise ValueError('Expression must have nonzero cells and genes')
    total = n_cells*n_genes
    zeros = negatives = nonfinite = 0
    minimum, maximum = float('inf'), float('-inf')
    sum_values = 0.
    for offset in range(0, n_cells, chunk_size):
        block = expression[offset:offset+chunk_size]
        if sparse.issparse(block):
            block = block.toarray()
        x = np.asarray(block, dtype=np.float64)
        zeros += int((x == 0).sum())
        negatives += int((x < 0).sum())
        nonfinite += int((~np.isfinite(x)).sum())
        finite = x[np.isfinite(x)]
        if finite.size:
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
            sum_values += float(finite.sum())
    if nonfinite == total:
        raise ValueError('All expression values are nonfinite')
    return ExpressionDiagnostics(n_cells, n_genes, minimum, maximum,
                                 sum_values/max(total-nonfinite, 1),
                                 zeros/total, negatives/total, nonfinite)


def ensure_expression_compatible(expression, *, require_nonnegative=False):
    diag = describe_expression(expression)
    if diag.nonfinite_count:
        raise ValueError(f'Expression contains {diag.nonfinite_count} nonfinite values')
    if require_nonnegative and diag.negative_fraction > 0:
        raise ValueError('Negative expression values in raw-counts input')
    return diag


def select_hvg_train_only(expression, train_rows, gene_names, *, top_genes=5000,
                          include_genes=()):





    from .data import dense_rows
    if top_genes < 1:
        raise ValueError('top_genes must be positive')
    selected_rows = np.asarray(train_rows, dtype=np.int64)
    if len(selected_rows) < 2:
        raise ValueError('Need at least two training cells for HVG selection')
    
    g = len(gene_names)
    sum_x = np.zeros(g, np.float64)
    sum_x2 = np.zeros(g, np.float64)
    for start in range(0, len(selected_rows), 2048):
        x = dense_rows(expression, selected_rows[start:start+2048]).astype(np.float64)
        sum_x += x.sum(0)
        sum_x2 += (x*x).sum(0)
    n = len(selected_rows)
    variance = np.maximum(0., sum_x2/n - (sum_x/n)**2)
    top = np.argsort(-variance)[:min(g, top_genes)].tolist()
    index = {g: i for i, g in enumerate(gene_names)}
    top.extend(index[g] for g in include_genes if g in index)
    return np.asarray(sorted(set(top)), dtype=np.int64)
