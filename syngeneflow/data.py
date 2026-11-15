





from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json

import numpy as np
import torch
from scipy import sparse

from .benchmarks.splits import (canonical as canonical_condition,
                                active_components as components,
                                condition_split, read_split_manifest, resolve_manifest)

CONTROL = "control"


def dense_rows(X, rows, genes=None) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    out = X[rows]
    if genes is not None:
        out = out[:, genes]
    if sparse.issparse(out):
        out = out.toarray()
    return np.asarray(out, dtype=np.float32)


def log1p_normalize(X, library_size=1e4):
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32).copy()
        depth = np.asarray(X.sum(axis=1)).ravel()
        scales = library_size / np.maximum(depth, 1.0)
        X = X.multiply(scales[:, None]).tocsr()
        X.data = np.log1p(X.data)
        return X
    X = np.asarray(X, np.float32)
    scale = library_size / np.maximum(X.sum(1, keepdims=True), 1.0)
    return np.log1p(X * scale).astype(np.float32)


def read_expression(path: str, condition_key: str, layer: str | None = None):

    path = Path(path)
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        for key in ("X", "conditions", "gene_names"):
            if key not in archive:
                raise KeyError(f"{path} lacks required field {key}")
        X = np.asarray(archive["X"], np.float32)
        cond = archive["conditions"].astype(str)
        genes = archive["gene_names"].astype(str)
    elif path.suffix.lower() == ".h5ad":
        try:
            import anndata
        except ImportError as exc:
            raise ImportError("Reading .h5ad requires 'anndata' (pip install -e .[data])") from exc
        adata = anndata.read_h5ad(path)
        if condition_key not in adata.obs:
            raise KeyError(f"Missing obs[{condition_key!r}]; columns={list(adata.obs)}")
        X = (adata.layers[layer] if layer else adata.X)
        X = X.astype(np.float32)
        cond = adata.obs[condition_key].astype(str).to_numpy()
        genes = adata.var_names.astype(str).to_numpy()
        del adata
    else:
        raise ValueError("Use .npz or .h5ad expression data")
    if len(set(genes)) != len(genes):
        raise ValueError("Gene names must be unique")
    cond = np.asarray([canonical_condition(c) for c in cond], dtype=str)
    if CONTROL not in cond:
        raise ValueError("At least one control cell is needed (control/ctrl labels)")
    if X.shape != (len(cond), len(genes)):
        raise ValueError("Expression matrix dimensions do not match metadata")
    return X, cond, genes


@dataclass
class ConditionSplit:
    train: list[str]
    validation: list[str]
    test: list[str]
    heldout_components: list[str]

    def as_dict(self):
        return dict(train=self.train, validation=self.validation,
                    test=self.test, heldout_components=self.heldout_components)


def create_split(conditions: Sequence[str], mode="additive", seed=42,
                 fold=0, holdout_count=12, test_fraction=0.3,
                 validation_fraction=0.1, supplied: dict | None = None) -> ConditionSplit:
    if supplied is not None:
        values = resolve_manifest(conditions, supplied)
    else:
        values = condition_split(conditions, mode=mode, seed=seed, fold=fold,
                                 holdout_count=holdout_count,
                                 test_fraction=test_fraction,
                                 validation_fraction=validation_fraction)
    return ConditionSplit(**values)


class SingleCellDataset:


    def __init__(self, X, conditions, gene_names, split: ConditionSplit,
                 *, n_genes=5000, stats_sample=4096, seed=42,
                 include_perturbed_genes=True, selected_gene_names=None):
        self.conditions = np.asarray([canonical_condition(c) for c in conditions])
        self.split = split
        self.rng = np.random.default_rng(seed)
        all_genes = np.asarray(gene_names, dtype=str)
        self.n_cells = len(self.conditions)
        train_all = np.where(np.isin(self.conditions, split.train))[0]
        if len(train_all) < 2:
            raise ValueError("At least 2 training cells are required")
        subset = self.rng.choice(train_all, min(stats_sample, len(train_all)), replace=False)
        sampled = dense_rows(X, subset)
        
        dispersion = sampled.var(axis=0)
        if selected_gene_names is not None:
            lookup = {str(name): i for i, name in enumerate(all_genes)}
            absent = [name for name in selected_gene_names if name not in lookup]
            if absent:
                raise ValueError(f'Configured gene list contains absent genes: {absent[:10]}')
            if len(selected_gene_names) != len(set(selected_gene_names)):
                raise ValueError('Gene list contains duplicates')
            selected = np.asarray([lookup[name] for name in selected_gene_names], dtype=np.int64)
        else:
            selected = np.argsort(-dispersion)[:min(n_genes, len(all_genes))]
            if include_perturbed_genes:
                gene_lookup = {g: i for i, g in enumerate(all_genes)}
                linked = [gene_lookup[p] for c in set(self.conditions) for p in components(c)
                          if p in gene_lookup]
                selected = np.unique(np.concatenate([selected, np.asarray(linked, np.int64)]))
            selected = np.sort(selected)
        self.gene_names = all_genes[selected]
        self.X = X[:, selected].tocsr() if sparse.issparse(X) else np.asarray(X[:, selected], np.float32)
        self.rows = {c: np.where(self.conditions == c)[0] for c in sorted(set(self.conditions))}
        self.control_rows = self.rows[CONTROL]
        training_ctrl = self.control_rows  
        if len(training_ctrl) < 2:
            raise ValueError("At least two training control cells required")
        ctrl_subset = self.rng.choice(training_ctrl, min(stats_sample, len(training_ctrl)), replace=False)
        self.control_std = dense_rows(self.X, ctrl_subset).std(0).astype(np.float32)
        self.control_mean = dense_rows(self.X, ctrl_subset).mean(0).astype(np.float32)
        self.train_conditions = [x for x in split.train if x != CONTROL and len(self.rows.get(x, []))]
        self.singles = [x for x in self.train_conditions if len(components(x)) == 1]
        if not self.train_conditions:
            raise ValueError("No training perturbation conditions")
        names = sorted(set(p for c in self.rows for p in components(c)))
        self.vocab = [CONTROL, *names]
        self.perturbation_to_id = {p: i for i, p in enumerate(self.vocab)}
        self.perturb_to_gene = [-1] * len(self.vocab)
        gene_lookup = {s: i for i, s in enumerate(self.gene_names)}
        for p in names:
            self.perturb_to_gene[self.perturbation_to_id[p]] = gene_lookup.get(p, -1)
        self.max_perturbations = max(2, max(len(components(c)) for c in self.rows))

    def condition_ids(self, condition: str, count: int) -> torch.Tensor:
        tokens = [self.perturbation_to_id[x] for x in components(condition)]
        tokens = (tokens + [0] * self.max_perturbations)[:self.max_perturbations]
        return torch.tensor(tokens, dtype=torch.long)[None].expand(count, -1).clone()

    def sample(self, batch_size: int, gene_indices: torch.Tensor, *, single=False,
               device="cpu", condition: str | None = None):
        available = self.singles if single else self.train_conditions
        if condition is None:
            if not available:
                return None
            condition = str(self.rng.choice(available))
        if condition not in self.rows:
            raise KeyError(condition)
        idx = gene_indices.cpu().numpy()
        source_rows = self.rng.choice(self.control_rows, batch_size, replace=True)
        target_rows = self.rng.choice(self.rows[condition], batch_size, replace=True)
        controls = torch.from_numpy(dense_rows(self.X, source_rows, idx)).to(device)
        targets = torch.from_numpy(dense_rows(self.X, target_rows, idx)).to(device)
        perturb = self.condition_ids(condition, batch_size).to(device)
        return controls, targets, perturb

    def reference_single_means(self, gene_indices=None):
        genes = np.arange(len(self.gene_names)) if gene_indices is None else gene_indices
        singles = {components(c)[0]: dense_rows(self.X, self.rows[c], genes).mean(0)
                   for c in self.singles}
        return singles


def prepare_data(input_path, *, condition_key="condition", layer=None,
                 transform="as_is", split_mode="additive", seed=42, fold=0,
                 n_genes=5000, split_manifest=None, holdout_count=12,
                 gene_list=None):
    X, conditions, genes = read_expression(input_path, condition_key, layer)
    if transform == "counts":
        X = log1p_normalize(X)
    elif transform != "as_is":
        raise ValueError("transform must be 'as_is' or 'counts'")
    spec = read_split_manifest(split_manifest, fold=fold) if split_manifest else None
    split = create_split(conditions, split_mode, seed, fold,
                         holdout_count=holdout_count, supplied=spec)
    selected_gene_names = None
    if gene_list is not None:
        source = Path(gene_list)
        if source.suffix.lower() == '.json':
            selected_gene_names = json.loads(source.read_text())
        else:
            selected_gene_names = [line.strip() for line in source.read_text().splitlines()
                                   if line.strip() and not line.lstrip().startswith('#')]
        if not isinstance(selected_gene_names, list) or not all(isinstance(x, str) for x in selected_gene_names):
            raise ValueError('Gene list must contain an ordered list of gene names')
    return SingleCellDataset(X, conditions, genes, split, n_genes=n_genes, seed=seed + fold,
                             selected_gene_names=selected_gene_names)
