
import numpy as np
from scipy.stats import spearmanr


def reference_additive_mean(condition, control_mean, single_means):

    parts = [x for x in str(condition).split("+") if x != "control"]
    if not parts or any(p not in single_means for p in parts):
        return None
    base = np.asarray(control_mean, dtype=np.float32)
    return base + sum((single_means[p] - base for p in parts),
                      np.zeros_like(base))


def synergy_mean_proxies(predicted, target, additive_reference, topk=50):

    predicted = np.asarray(predicted, np.float32).mean(0)
    target = np.asarray(target, np.float32).mean(0)
    ref = np.asarray(additive_reference, np.float32)
    pred_s, true_s = predicted - ref, target - ref
    count = min(topk, len(true_s))
    if count <= 0:
        raise ValueError("No genes available")
    top_pred = set(np.argsort(-np.abs(pred_s))[:count])
    top_true = set(np.argsort(-np.abs(true_s))[:count])
    if np.std(pred_s) > 1e-8 and np.std(true_s) > 1e-8:
        pearson = float(np.corrcoef(pred_s, true_s)[0, 1])
        spearman = float(spearmanr(np.abs(pred_s), np.abs(true_s)).statistic)
    else:
        pearson, spearman = float("nan"), float("nan")
    return {
        "additive_reference_MSE_proxy": float(np.mean((ref - target) ** 2)),
        "synergy_MSE_proxy": float(np.mean((pred_s - true_s) ** 2)),
        "synergy_Pearson_proxy": pearson,
        "synergy_abs_DE_Spearman_proxy": spearman,
        "synergy_nonadditive_recall_at50_proxy": len(top_pred & top_true) / count,
    }
