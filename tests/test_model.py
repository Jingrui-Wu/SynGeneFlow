
import numpy as np
import torch
import pytest

from syngeneflow.graph import (topk_coexpression_neighbors,
                                      induced_graph_mask, sample_connected_subset)
from syngeneflow.losses import multikernel_mmd2, synergy_flow_losses
from syngeneflow.model import SynGeneFlow


def setup_case():
    torch.manual_seed(17)
    X = torch.randn(25, 12)
    neighbors = topk_coexpression_neighbors(X.numpy(), k=3, chunk_size=4)
    indices = sample_connected_subset(neighbors, 8)
    mask = induced_graph_mask(neighbors, indices)
    model = SynGeneFlow(12, 7, control_id=0, perturb_to_gene=[-1, 1, 2, 3, -1, -1, -1],
                        hidden=24, heads=4, gene_layers=1, velocity_layers=1, edge_hidden=12)
    source = torch.rand(5, 8)
    target = torch.rand(5, 8)
    perturbation = torch.tensor([[1, 2]]).expand(5, -1).clone()
    single_ids = torch.tensor([[1, 0]]).expand(5, -1).clone()
    std = torch.rand(12) + 0.1
    return model, neighbors, indices, mask, source, target, perturbation, single_ids, std


def test_topk_and_graph():
    model, neighbors, indices, mask, *_ = setup_case()
    assert neighbors.shape == (12, 3)
    assert mask.shape == (8, 8)
    assert mask.diagonal().all()
    assert torch.equal(mask, mask.T)
    assert mask.sum() > 8
    assert not (neighbors == torch.arange(12)[:, None]).any()
    assert induced_graph_mask(neighbors, indices, disable_mask=True).all()


def test_additive_no_interaction_leakage():
    m, _, i, _, c, _, _, _, _ = setup_case()
    c0 = torch.zeros((5, 2), dtype=torch.long)
    a = torch.tensor([[1, 0]]).expand(5, -1)
    b = torch.tensor([[2, 0]]).expand(5, -1)
    ab = torch.tensor([[1, 2]]).expand(5, -1)
    with torch.no_grad():
        base = m.additive_prior(c, c0, i)
        d_a = m.additive_prior(c, a, i) - base
        d_b = m.additive_prior(c, b, i) - base
        d_ab = m.additive_prior(c, ab, i) - base
        assert torch.allclose(base, c, atol=1e-6)
        assert torch.allclose(d_a + d_b, d_ab, atol=1e-6)


def test_losses_and_backward():
    m, _, i, mask, c, y, p, single, std = setup_case()
    stats = synergy_flow_losses(m, c, y, p, i, mask, std,
                                single_batch=(c, y, single))
    for key in ('total','flow','endpoint_mmd','synergy_mmd','single'):
        assert stats[key].ndim == 0, key
        assert torch.isfinite(stats[key]), key
    stats['total'].backward()
    assert m.single_effect_decoder[0].weight.grad is not None
    assert m.gene_encoder.blocks[0]['attn'].edge_mlp[0].weight.grad is not None
    assert m.velocity_head[-1].weight.grad is not None
    assert m.perturbation_embedding.weight.grad is not None


def test_dynamic_deformation_changes_with_condition():
    m, _, i, mask, c, y, p, _, std = setup_case()
    m.eval()
    p2 = torch.tensor([[3, 0]]).expand(5, -1)
    with torch.no_grad():
        v1 = m.velocity(y, torch.zeros(5), c, p, i, mask)
        v2 = m.velocity(y, torch.zeros(5), c, p2, i, mask)
    assert v1.shape == y.shape
    assert not torch.allclose(v1, v2)


def test_inference_determinism_and_stochasticity():
    m, _, i, mask, c, y, p, _, std = setup_case()
    m.eval()
    out1 = m.generate(c, p, i, mask, std, steps=2, tau=0)
    out2 = m.generate(c, p, i, mask, std, steps=2, tau=0)
    assert torch.allclose(out1, out2, atol=1e-6)
    assert (out1 >= 0).all()
    torch.manual_seed(1)
    stoch1 = m.generate(c, p, i, mask, std, steps=2, tau=0.7)
    torch.manual_seed(2)
    stoch2 = m.generate(c, p, i, mask, std, steps=2, tau=0.7)
    assert not torch.allclose(stoch1, stoch2)


def test_ablation_path():
    m, _, i, mask, c, y, p, single, std = setup_case()
    s = synergy_flow_losses(m, c, y, p, i, mask, std,
                            single_batch=(c, y, single),
                            ablate=['additive_prior','synergy_mmd','endpoint_mmd','differential_attention'])
    assert s['synergy_mmd'].item() == 0
    assert s['single'].item() == 0
    assert torch.isfinite(s['total'])
    out = m.generate(c, p, i, mask, std, steps=2,
                     ablate=['additive_prior','differential_attention','gene_mask'])
    assert out.shape == y.shape


def test_mmd_detects_shift_and_batch_one():
    torch.manual_seed(3)
    x = torch.randn(36, 8)
    y = torch.randn(36, 8)
    y_shifted = y + 4
    assert multikernel_mmd2(x, y_shifted) > multikernel_mmd2(x, y)
    assert multikernel_mmd2(x[:1], y[:1]).item() == 0


def test_synergy_signature_proxies():
    from syngeneflow.evaluation import (reference_additive_mean,
                                              synergy_mean_proxies)
    base = np.full(60, 1.0)
    singles = {'A': base + np.linspace(0, 0.5, 60),
               'B': base + np.linspace(0.2, 0, 60)}
    ref = reference_additive_mean('A+B', base, singles)
    assert np.allclose(ref, singles['A'] + singles['B'] - base)
    assert reference_additive_mean('A+C', base, singles) is None
    target = ref + np.arange(60)*0.001
    metrics = synergy_mean_proxies(target[None], target[None], ref)
    assert metrics['synergy_MSE_proxy'] == 0.
    assert np.isclose(metrics['synergy_Pearson_proxy'], 1.0)
    assert np.isclose(metrics['synergy_nonadditive_recall_at50_proxy'], 1.0)
