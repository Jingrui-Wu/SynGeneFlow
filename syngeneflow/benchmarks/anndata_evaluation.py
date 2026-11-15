
from __future__ import annotations
from pathlib import Path
import numpy as np


def annotated_population(controls, generated, observed, labels, gene_names):





    import anndata as ad
    import pandas as pd
    cx = np.asarray(controls, np.float32)
    p = np.asarray(generated, np.float32)
    y = np.asarray(observed, np.float32)
    if not (cx.shape[1] == p.shape[1] == y.shape[1] == len(gene_names)):
        raise ValueError('Gene order differs between real/predicted data')
    if len(labels) != 2:
        raise ValueError('Expected two label lists: generated and observed')
    pl, yl = labels
    if len(pl) != len(p) or len(yl) != len(y):
        raise ValueError('Prediction/observation labels do not match cell counts')
    var = pd.DataFrame(index=list(gene_names))
    pred = ad.AnnData(X=np.vstack([cx, p]),
        obs=pd.DataFrame({'perturbation': ['control']*len(cx)+list(pl)}), var=var.copy())
    real = ad.AnnData(X=np.vstack([cx, y]),
        obs=pd.DataFrame({'perturbation': ['control']*len(cx)+list(yl)}), var=var.copy())
    return pred, real


def export_anndata_pair(controls, generated, observed, labels, gene_names, path):
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    pred, real = annotated_population(controls, generated, observed, labels, gene_names)
    pred.write_h5ad(folder/'pred.h5ad')
    real.write_h5ad(folder/'real.h5ad')
    return folder/'pred.h5ad', folder/'real.h5ad'


def run_cell_evaluation(predicted_h5ad, observed_h5ad, output_dir, *, threads=8):

    import anndata as ad
    try:
        from cell_eval import MetricsEvaluator
    except ImportError as exc:
        raise ImportError('Install a compatible cell_eval version for benchmark scoring') from exc
    pred = ad.read_h5ad(predicted_h5ad)
    real = ad.read_h5ad(observed_h5ad)
    if list(pred.var_names) != list(real.var_names):
        raise ValueError('Real and predicted AnnData gene ordering differs')
    evaluator = MetricsEvaluator(adata_pred=pred, adata_real=real,
                                 control_pert='control', pert_col='perturbation',
                                 num_threads=threads)
    per_condition, aggregated = evaluator.compute()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_condition.write_csv(out/'results.csv')
    aggregated.write_csv(out/'agg_results.csv')
    return per_condition, aggregated
