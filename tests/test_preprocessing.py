import numpy as np
import pytest
from scipy.sparse import csr_matrix
from syngeneflow.preprocessing import (describe_expression,
                                        ensure_expression_compatible,
                                        select_hvg_train_only)
from syngeneflow.data import SingleCellDataset, create_split


def test_diagnostics_dense_sparse_identical():
    X = np.eye(6).astype(np.float32)
    dense = describe_expression(X)
    sparse = describe_expression(csr_matrix(X))
    assert dense == sparse
    assert dense.zero_fraction == 30/36
    assert dense.nonfinite_count == 0


def test_invalid_expression_rejected():
    X = np.ones((10,6),dtype=np.float32)
    X[1,3] = np.nan
    with pytest.raises(ValueError):
        ensure_expression_compatible(X)
    X[1,3] = -2.
    with pytest.raises(ValueError):
        ensure_expression_compatible(X,require_nonnegative=True)


def test_train_only_hvg_ignores_heldout_outlier():
    X=np.zeros((12,4),dtype=np.float32)
    X[:8,2] = np.arange(8)
    X[8:,3] = np.arange(4)*1000
    genes=['a','b','c','d']
    idx=select_hvg_train_only(X,np.arange(8),genes,top_genes=1)
    assert idx.tolist() == [2]
    forced=select_hvg_train_only(X,np.arange(8),genes,top_genes=1,include_genes=['a'])
    assert forced.tolist() == [0,2]


def test_fixed_gene_order_is_respected():
    conditions = ['control']*8 + ['A']*8 + ['B']*8 + ['A+B']*8
    X=np.random.default_rng(7).random((32,10)).astype(np.float32)
    genes=[f'G{i}' for i in range(10)]
    split=create_split(conditions,mode='additive')
    ds=SingleCellDataset(X,conditions,genes,split,n_genes=2,selected_gene_names=['G7','G1','G4'])
    assert ds.gene_names.tolist() == ['G7','G1','G4']
