
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ResidualPathBatch:
    anchor: torch.Tensor
    source: torch.Tensor
    intermediate: torch.Tensor
    time: torch.Tensor
    velocity: torch.Tensor
    biological_synergy: torch.Tensor
    transport_residual: torch.Tensor


def sample_residual_path(anchor: torch.Tensor, target: torch.Tensor,
                         control_std: torch.Tensor, tau: float = 0.1,
                         generator: torch.Generator | None = None,
                         time: torch.Tensor | None = None) -> ResidualPathBatch:
    if anchor.shape != target.shape:
        raise ValueError('Anchor/target expression dimensions must match')
    if anchor.ndim != 2:
        raise ValueError('Expected 2D batch of expression vectors')
    if control_std.shape != (anchor.shape[1],):
        raise ValueError('Std vector must correspond to the selected gene dimension')
    if tau < 0:
        raise ValueError('tau must be nonnegative')
    noise = torch.randn(anchor.shape, dtype=anchor.dtype, device=anchor.device, generator=generator)
    source = anchor + noise * float(tau) * control_std[None]
    if time is None:
        time = torch.rand((anchor.shape[0], 1), device=anchor.device,
                          dtype=anchor.dtype, generator=generator)
    elif time.shape != (anchor.shape[0], 1):
        raise ValueError('time must be [B,1]')
    xt = (1.0 - time) * source + time * target
    transported = target - source
    synergy = target - anchor
    return ResidualPathBatch(anchor, source, xt, time, transported, synergy, transported)
