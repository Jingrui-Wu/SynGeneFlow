



from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .splits import canonical, active_components, read_split_manifest, resolve_manifest
from .metrics import (population_response_metrics, population_synergy_metrics,
                      compute_population_mmd)
from .bootstrap import condition_bootstrap
from ..data import read_expression, dense_rows, log1p_normalize


def load_population(path, *, condition_key='condition', layer=None, transform='as_is'):

    path = Path(path)
    if path.suffix == '.npz':
        archive = np.load(path, allow_pickle=False)
        key = 'condition' if 'condition' in archive else 'conditions'
        if not all(k in archive for k in ('X','gene_names')) or key not in archive:
            raise ValueError('Population archive needs X, gene_names and conditions')
        X, condition, genes = archive['X'], archive[key].astype(str), archive['gene_names'].astype(str)
        X = np.asarray(X, dtype=np.float32)
        condition = np.asarray([canonical(x) for x in condition], dtype=str)
    else:
        X, condition, genes = read_expression(str(path), condition_key, layer)
    if transform == 'counts':
        X = log1p_normalize(X)
    elif transform != 'as_is':
        raise ValueError('Unsupported transform')
    return X, condition, genes


def align_expression(X, from_genes, target_genes):
    lookup = {g: i for i, g in enumerate(from_genes)}
    missing = [g for g in target_genes if g not in lookup]
    if missing:
        raise ValueError(f'Prediction and observation gene sets differ: {missing[:8]}')
    selected = np.asarray([lookup[g] for g in target_genes],dtype=int)
    return X[:, selected]


def evaluate_populations(predicted_X, predicted_conditions, observed_X,
                         observed_conditions, gene_names, *, train_conditions=None,
                         max_cells=512, seed=42, compute_mmd=True, top_de=20):





    if max_cells < 2:
        raise ValueError('max_cells must be >=2')
    gene_names = list(gene_names)
    if predicted_X.shape[1] != observed_X.shape[1] or observed_X.shape[1] != len(gene_names):
        raise ValueError('Mismatched expression/gene dimensions')
    rng = np.random.default_rng(seed)
    predicted_conditions = np.asarray([canonical(x) for x in predicted_conditions])
    observed_conditions = np.asarray([canonical(x) for x in observed_conditions])
    ctrl_rows = np.where(observed_conditions == 'control')[0]
    if len(ctrl_rows) < 2:
        raise ValueError('Need at least two observed control cells')
    control = dense_rows(observed_X, ctrl_rows[:max_cells])
    control_mean = control.mean(0)
    if train_conditions is None:
        
        train_conditions = {'control'}
    else:
        train_conditions = {canonical(x) for x in train_conditions}
    observed_groups = {c: np.where(observed_conditions==c)[0]
                       for c in np.unique(observed_conditions)}
    means = {}
    for condition in train_conditions:
        if len(active_components(condition)) == 1 and condition in observed_groups:
            means[active_components(condition)[0]] = dense_rows(
                observed_X, observed_groups[condition]).mean(0)
    report = []
    for condition in sorted(set(predicted_conditions)-{'control'}):
        if condition not in observed_groups:
            continue
        prow = np.flatnonzero(predicted_conditions == condition)
        trow = observed_groups[condition]
        if len(prow) < 2 or len(trow) < 2:
            continue
        pids = rng.choice(prow, min(len(prow),max_cells), replace=False)
        tids = rng.choice(trow, min(len(trow),max_cells), replace=False)
        pred = dense_rows(predicted_X, pids)
        obs = dense_rows(observed_X, tids)
        result = {'condition': condition,
                  **population_response_metrics(pred, obs, control, top_de=top_de)}
        if compute_mmd:
            result['expression_MMD2'] = compute_population_mmd(pred, obs)
        parts = active_components(condition)
        if parts and all(c in means for c in parts):
            additive = control_mean + sum((means[c]-control_mean for c in parts),
                                          np.zeros_like(control_mean))
            result.update(population_synergy_metrics(pred,obs,additive))
        report.append(result)
    return report


def aggregate_conditions(rows, *, confidence=.95, bootstrap_reps=1000, seed=42):
    keys = sorted({k for item in rows for k,v in item.items()
                   if isinstance(v,(int,float,np.integer,np.floating))})
    summary = {}
    for i,key in enumerate(keys):
        values = [float(row.get(key,float('nan'))) for row in rows]
        summary[key] = condition_bootstrap(values, confidence=confidence,
                                            replications=bootstrap_reps,
                                            seed=seed+i).as_dict()
    return summary


def arguments(argv=None):
    p = argparse.ArgumentParser(description='SynGeneFlow held-out population assessment')
    p.add_argument('--predictions',required=True,help='Generated .npz or .h5ad')
    p.add_argument('--observations',required=True,help='Held-out evaluation .npz or .h5ad')
    p.add_argument('--split-manifest',default=None,
                   help='Optional training/validation/test condition metadata')
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--condition-key',default='condition')
    p.add_argument('--observed-layer',default=None)
    p.add_argument('--transform',choices=['as_is','counts'],default='as_is')
    p.add_argument('--max-cells',type=int,default=512)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    predicted, pred_conditions, pred_genes = load_population(args.predictions)
    observed, obs_conditions, obs_genes = load_population(
        args.observations, condition_key=args.condition_key,
        layer=args.observed_layer, transform=args.transform)
    
    
    observed = align_expression(observed,obs_genes,pred_genes)
    train_conditions = None
    if args.split_manifest:
        manifest = read_split_manifest(args.split_manifest,fold=args.fold)
        resolved = resolve_manifest(obs_conditions,manifest)
        train_conditions = resolved['train']
    results = evaluate_populations(predicted,pred_conditions,observed,obs_conditions,
                                   pred_genes,train_conditions=train_conditions,
                                   max_cells=args.max_cells,seed=args.seed)
    aggregate = aggregate_conditions(results,bootstrap_reps=args.bootstrap,seed=args.seed)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if results:
        keys = sorted({k for row in results for k in row})
        with (out/'per_condition.csv').open('w',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    (out/'summary.json').write_text(json.dumps({'metrics':aggregate,
                                                'n_conditions':len(results)},indent=2,allow_nan=False))
    print(f'Assessed {len(results)} held-out perturbation populations')


if __name__ == '__main__':
    main()
