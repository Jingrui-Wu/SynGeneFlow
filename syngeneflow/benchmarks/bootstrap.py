






from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    n_conditions: int
    confidence: float
    replications: int

    def as_dict(self):
        def finite_or_none(value):
            return float(value) if np.isfinite(value) else None
        return dict(estimate=finite_or_none(self.estimate),
                    lower=finite_or_none(self.lower),
                    upper=finite_or_none(self.upper),
                    n_conditions=self.n_conditions, confidence=self.confidence,
                    replications=self.replications)


def condition_bootstrap(values, *, confidence=.95, replications=1000, seed=42):

    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if not 0 < confidence < 1:
        raise ValueError('Confidence level must be between 0 and 1')
    if replications < 2:
        raise ValueError('At least two bootstrap replications are required')
    if len(a) == 0:
        return BootstrapInterval(float('nan'),float('nan'),float('nan'),0,
                                 confidence,replications)
    if len(a) == 1:
        return BootstrapInterval(float(a[0]),float('nan'),float('nan'),1,
                                 confidence,replications)
    rng = np.random.default_rng(seed)
    means = np.empty(replications, dtype=np.float64)
    for i in range(replications):
        means[i] = a[rng.integers(0,len(a),size=len(a))].mean()
    lo = 100*(1-confidence)/2
    hi = 100*(1+confidence)/2
    return BootstrapInterval(float(a.mean()),float(np.percentile(means,lo)),
                             float(np.percentile(means,hi)),len(a),
                             confidence,replications)


def paired_condition_comparison(ours, reference, *, confidence=.95,
                                replications=1000, seed=42):





    a, b = np.asarray(ours,float), np.asarray(reference,float)
    if a.shape != b.shape:
        raise ValueError('Reference and candidate must align by condition')
    return condition_bootstrap(a-b, confidence=confidence,
                               replications=replications, seed=seed)
