




from __future__ import annotations
from contextlib import contextmanager
from collections import OrderedDict
import torch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError('EMA decay must satisfy 0 <= decay < 1')
        self.decay = float(decay)
        self.updates = 0
        self.shadow = OrderedDict((n, p.detach().clone())
                                  for n, p in model.state_dict().items())

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        
        effective_decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        current = model.state_dict()
        if self.shadow.keys() != current.keys():
            raise ValueError('EMA model architecture changed during training')
        for name, current_tensor in current.items():
            if torch.is_floating_point(self.shadow[name]):
                self.shadow[name].mul_(effective_decay).add_(current_tensor.detach(),
                                                              alpha=1.0 - effective_decay)
            else:
                self.shadow[name].copy_(current_tensor.detach())

    def state_dict(self):
        return {'decay': self.decay, 'updates': self.updates,
                'shadow': OrderedDict((k, v.clone()) for k, v in self.shadow.items())}

    def load_state_dict(self, state):
        if self.shadow.keys() != state['shadow'].keys():
            raise ValueError('EMA state does not match model')
        self.decay = float(state['decay'])
        self.updates = int(state['updates'])
        for key, val in state['shadow'].items():
            if self.shadow[key].shape != val.shape:
                raise ValueError(f'Invalid EMA shape: {key}')
            self.shadow[key].copy_(val)

    @contextmanager
    def average_parameters(self, model):

        original = OrderedDict((k, v.detach().clone())
                               for k, v in model.state_dict().items())
        try:
            model.load_state_dict(self.shadow, strict=True)
            yield model
        finally:
            model.load_state_dict(original, strict=True)
