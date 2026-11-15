





from __future__ import annotations
from typing import Callable
import torch

Tensor = torch.Tensor
Velocity = Callable[[Tensor, Tensor], Tensor]


def integrate(source: Tensor, field: Velocity, *, steps: int = 20,
              method: str = 'euler', return_trajectory: bool = False):

    if steps < 1:
        raise ValueError('steps must be >= 1')
    if method not in {'euler', 'heun', 'rk4'}:
        raise ValueError(f'Unsupported method: {method}')
    x = source
    snapshots = [source] if return_trajectory else None
    dt = 1.0 / steps

    def at(z, instant):
        t = torch.full((z.shape[0],), instant, dtype=z.dtype, device=z.device)
        v = field(z, t)
        if v.shape != z.shape:
            raise ValueError(f'Velocity shape {v.shape} differs from state {z.shape}')
        return v

    for k in range(steps):
        t = k * dt
        k1 = at(x, t)
        if method == 'euler':
            x = x + dt * k1
        elif method == 'heun':
            k2 = at(x + dt * k1, (k + 1) * dt)
            x = x + 0.5 * dt * (k1 + k2)
        else:
            k2 = at(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = at(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = at(x + dt * k3, (k + 1) * dt)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if snapshots is not None:
            snapshots.append(x)
    return torch.stack(snapshots, 0) if return_trajectory else x
