
import json
import numpy as np
import pytest
import torch
from syngeneflow.transport.interpolation import sample_residual_path
from syngeneflow.transport.integrators import integrate
from syngeneflow.benchmarks.splits import (canonical, condition_split, resolve_manifest,
                                            read_split_manifest)
from syngeneflow.benchmarks.metrics import (population_response_metrics,
                                             population_synergy_metrics,
                                             nonadditive_recall)
from syngeneflow.benchmarks.response_signatures import build_signature, rank_nonadditive_genes
from syngeneflow.benchmarks.reproducibility import split_audit


def test_residual_path_formula():
    control = torch.ones(5, 6)
    target = torch.ones(5, 6) * 4
    t = torch.full((5, 1), 0.3)
    path = sample_residual_path(control, target, torch.ones(6), tau=0., time=t)
    assert torch.equal(path.source, control)
    assert torch.allclose(path.intermediate, torch.full_like(control, 1.9))
    assert torch.equal(path.velocity, target-control)
    assert torch.equal(path.biological_synergy, target-control)


def test_stochastic_transport_residual_is_distinct_from_synergy():
    anchor = torch.randn(6, 8)
    target = torch.randn(6, 8)
    path = sample_residual_path(anchor, target, torch.ones(8), tau=1.)
    assert torch.allclose(path.velocity, target-path.source)
    assert torch.allclose(path.biological_synergy, target-anchor)
    assert not torch.allclose(path.velocity, path.biological_synergy)


@pytest.mark.parametrize('method', ['euler','heun','rk4'])
def test_constant_velocity_integration(method):
    x0 = torch.zeros(3, 7)
    out = integrate(x0, lambda x,t: torch.full_like(x, 2.), steps=5, method=method)
    assert torch.allclose(out, torch.full_like(out, 2.), atol=1e-6)
    traj = integrate(x0, lambda x,t: torch.full_like(x, 2.), steps=5,
                     method=method, return_trajectory=True)
    assert traj.shape == (6, 3, 7)
    assert torch.allclose(traj[-1], out)


def test_linear_velocity_convergence():
    x0 = torch.ones(2, 1)
    exact = np.e
    errors = {}
    for mode in ('euler','heun','rk4'):
        predicted = integrate(x0, lambda x,t:x, steps=4, method=mode)
        errors[mode] = abs(float(predicted.mean()) - exact)
    assert errors['rk4'] < errors['heun'] < errors['euler']


def test_drug_curated_split_includes_single_conditions():
    conditions = ['control', 'Panobinostat+Crizotinib','control+Alvespimycin',
                  'control+Dacinostat', 'Curcumin', 'SRT2104', 'Sorafenib']
    split = condition_split(conditions, mode='drug', seed=2)
    assert 'Alvespimycin' in split['test']
    assert 'Dacinostat' in split['test']
    assert 'Crizotinib+Panobinostat' in split['test']
    assert not (set(split['test']) & set(split['train']))


def test_legacy_condition_fold_file_supported(tmp_path):
    path = tmp_path/'fold.json'
    path.write_text(json.dumps([{'train': ['B+C'], 'test':['A+B']}]))
    spec = read_split_manifest(path, fold=0)
    fixed = resolve_manifest(['control','A','B','C','A+B','B+C'],spec)
    assert fixed['train'] == ['control','A','B','B+C','C']
    assert fixed['test'] == ['A+B']


def test_unseen_split_combinatorial_exclusion():
    conditions = ['control', 'A', 'B', 'C', 'D', 'A+B', 'B+C', 'C+D', 'A+D']
    split = condition_split(conditions,mode='unseen',seed=27,holdout_count=1)
    hold = set(split['heldout_components'])
    assert len(hold) == 1
    for cond in split['train']:
        assert not hold.intersection(set(canonical(cond).split('+')))
    split_audit(split, {canonical(c) for c in conditions})


def test_response_and_synergy_metrics():
    control = np.ones((20, 90))
    single = {'A': np.ones(90) + np.arange(90)*0.01,
              'B': np.ones(90) + np.arange(90)[::-1]*0.01}
    expected_additive = single['A']+single['B']-control.mean(0)
    observed = np.tile(expected_additive+np.linspace(0,.2,90), (16,1))
    pred = observed.copy()
    signature = build_signature('A+B', control, observed, single)
    assert signature.has_observed_single_reference()
    assert len(rank_nonadditive_genes(signature,[f'G{i}' for i in range(90)]))==50
    m = population_response_metrics(pred, observed, control)
    assert np.isclose(m['mean_MSE'],0.)
    synergy = population_synergy_metrics(pred, observed, expected_additive)
    assert np.isclose(synergy['synergy_MSE'],0.)
    assert np.isclose(synergy['nonadditive_recall_at50'],1.)


def test_no_spurious_reference_for_unobserved_single():
    c=np.ones((6,5))
    y=np.ones((8,5))*3
    assert not build_signature('A+U',c,y,{'A':np.ones(5)*2}).has_observed_single_reference()
