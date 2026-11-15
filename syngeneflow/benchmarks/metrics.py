





from __future__ import annotations
import numpy as np
from scipy.stats import rankdata, spearmanr


def _array(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError('Populations must be matrices [cells, genes]')
    if not np.isfinite(a).all():
        raise ValueError('Expressions must be finite')
    return a


def safe_pearson(x, y) -> float:
    a = np.asarray(x, np.float64).ravel()
    b = np.asarray(y, np.float64).ravel()
    if len(a) < 2 or len(a) != len(b) or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(x, y) -> float:
    a = np.asarray(x, np.float64).ravel()
    b = np.asarray(y, np.float64).ravel()
    if len(a) < 2 or len(a) != len(b):
        return float('nan')
    return float(spearmanr(a, b).statistic)


def topk_indices(x, k: int) -> np.ndarray:
    x = np.asarray(x)
    if k <= 0:
        raise ValueError('k must be positive')
    return np.argsort(-np.abs(x), kind='stable')[:min(k, len(x))]


def nonadditive_recall(pred_residual, true_residual, k=50) -> float:
    pred = set(topk_indices(pred_residual, k).tolist())
    true = set(topk_indices(true_residual, k).tolist())
    return len(pred & true) / max(len(true), 1)


def population_response_metrics(predicted, observed, control, *, top_de=20):
    pred, obs, ctrl = _array(predicted), _array(observed), _array(control)
    if not (pred.shape[1] == obs.shape[1] == ctrl.shape[1]):
        raise ValueError('All populations must measure the same ordered genes')
    p, y, c = pred.mean(0), obs.mean(0), ctrl.mean(0)
    dp, dy = p-c, y-c
    active = topk_indices(dy, top_de)
    return {
        'mean_L2': float(np.linalg.norm(p-y)),
        'mean_MSE': float(np.mean((p-y)**2)),
        'mean_MAE': float(np.mean(np.abs(p-y))),
        'Pearson_delta': safe_pearson(dp, dy),
        'Pearson_delta_top20': safe_pearson(dp[active], dy[active]),
        'DE_Spearman_top20': safe_spearman(dp[active], dy[active]),
        'n_predicted': len(pred),
        'n_observed': len(obs),
    }


def population_synergy_metrics(predicted, observed, additive_reference, *, k=50):
    p, y = _array(predicted), _array(observed)
    a = np.asarray(additive_reference, np.float64)
    if p.shape[1] != len(a) or y.shape[1] != len(a):
        raise ValueError('Incorrect additive reference length')
    
    sp = p.mean(0) - a
    sy = y.mean(0) - a
    de = topk_indices(sy, k)
    return {
        'synergy_MSE': float(np.mean((sp - sy)**2)),
        'synergy_Pearson': safe_pearson(sp, sy),
        'synergy_DE_Spearman': safe_spearman(sp[de], sy[de]),
        'nonadditive_recall_at50': nonadditive_recall(sp, sy, k),
    }


def compute_population_mmd(predicted, observed, scales=(0.5, 1., 2., 4.)) -> float:
    import torch
    from ..losses import multikernel_mmd2
    a = torch.as_tensor(_array(predicted), dtype=torch.float32)
    b = torch.as_tensor(_array(observed), dtype=torch.float32)
    return float(multikernel_mmd2(a, b).detach())
