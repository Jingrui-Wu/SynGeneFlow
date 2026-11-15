
from __future__ import annotations
from pathlib import Path
import os
import tempfile
import torch


def atomic_torch_save(payload, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp = tempfile.mkstemp(prefix='.partial-', suffix='.pt', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def checkpoint_summary(payload):

    required = ('step', 'model', 'config', 'gene_names', 'vocab', 'split', 'graph')
    absent = [key for key in required if key not in payload]
    if absent:
        raise KeyError(f'Missing checkpoint fields: {absent}')
    return {
        'step': int(payload['step']),
        'n_genes': len(payload['gene_names']),
        'n_perturbations': len(payload['vocab']),
        'split_counts': {key: len(payload['split'].get(key, []))
                         for key in ('train', 'validation', 'test')},
        'model_parameter_tensors': len(payload['model']),
        'has_optimizer': 'optimizer' in payload,
        'has_ema': payload.get('ema') is not None,
    }
