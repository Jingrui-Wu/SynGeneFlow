
import torch
import torch.nn.functional as F
from .transport.interpolation import sample_residual_path


def multikernel_mmd2(x, y, scales=(0.5, 1., 2., 4.)):






    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError('Expected expression batches [B,G] with same G')
    if x.shape[0] < 2 or y.shape[0] < 2:
        
        return (x.sum() + y.sum()) * 0.
    pooled = torch.cat([x.detach().float(), y.detach().float()])
    with torch.no_grad():
        d = torch.pdist(pooled, p=2).square()
        valid = d[d > 1e-12]
        median_d2 = valid.median() if valid.numel() else pooled.new_tensor(1.)
        median_d2 = median_d2.clamp_min(1e-8)
    dxx = torch.cdist(x.float(), x.float()).square()
    dyy = torch.cdist(y.float(), y.float()).square()
    dxy = torch.cdist(x.float(), y.float()).square()
    n, m = x.shape[0], y.shape[0]
    result = 0.
    for scale in scales:
        denominator = (median_d2 * float(scale)).clamp_min(1e-8)
        kxx = torch.exp(-dxx / denominator)
        kyy = torch.exp(-dyy / denominator)
        kxy = torch.exp(-dxy / denominator)
        xx = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
        yy = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
        result = result + xx + yy - 2. * kxy.mean()
    return result / len(scales)


def synergy_flow_losses(model, control, target, pert_ids, gene_indices, graph_mask,
                        control_std, *, tau=0.1, lambda_end=0.2,
                        lambda_syn=0.2, lambda_single=1., single_batch=None,
                        ablate=(), generator=None):






    ablate = set(ablate)
    if control.shape != target.shape:
        raise ValueError('control and target must both have [B,selected_G] shape')
    if 'additive_prior' in ablate:
        anchor = torch.zeros_like(control)
        z0 = torch.randn_like(control)  
    else:
        anchor = model.additive_prior(control, pert_ids, gene_indices)
        path = sample_residual_path(anchor, target, control_std[gene_indices],
                                    tau=tau, generator=generator)
        z0 = path.source
    if 'additive_prior' in ablate:
        t = torch.rand(control.shape[0], 1, device=control.device, dtype=control.dtype, generator=generator)
        xt = (1. - t) * z0 + t * target
        velocity_target = target - z0
    else:
        t, xt, velocity_target = path.time, path.intermediate, path.velocity
    if 'residual_flow' in ablate:
        velocity_pred = torch.zeros_like(target)
        fm = anchor.sum() * 0.
    else:
        velocity_pred = model.velocity(xt, t.squeeze(-1), control, pert_ids,
                                       gene_indices, graph_mask,
                                       use_deformation='deformation' not in ablate,
                                       use_differential='differential_attention' not in ablate)
        fm = F.mse_loss(velocity_pred, velocity_target)
    
    
    endpoint = anchor if 'residual_flow' in ablate else xt + (1. - t) * velocity_pred
    predicted_synergy = endpoint - anchor
    true_synergy = target - anchor  
    zero = fm.new_zeros(())
    end_mmd = (multikernel_mmd2(endpoint, target) if 'endpoint_mmd' not in ablate and lambda_end else zero)
    syn_mmd = (multikernel_mmd2(predicted_synergy, true_synergy) if 'synergy_mmd' not in ablate and lambda_syn else zero)
    single = zero
    if single_batch is not None and 'single_consistency' not in ablate and 'additive_prior' not in ablate:
        single_control, single_target, single_pert = single_batch
        assert ((single_pert != model.control_id).sum(-1) == 1).all(), 'L_single requires observed singles'
        single_pred = model.additive_prior(single_control, single_pert, gene_indices)
        single = F.mse_loss(single_pred, single_target)
    total = fm + lambda_end * end_mmd + lambda_syn * syn_mmd + lambda_single * single
    return {'total': total, 'flow': fm, 'endpoint_mmd': end_mmd,
            'synergy_mmd': syn_mmd, 'single': single,
            'anchor': anchor, 'endpoint': endpoint, 'synergy': predicted_synergy}
