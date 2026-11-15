
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def array_hash(a) -> str:
    arr = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def split_audit(split: dict, conditions):
    universe = set(conditions)
    groups = {key: set(split.get(key, [])) for key in ('train', 'validation', 'test')}
    for a, b in (('train','validation'),('train','test'),('validation','test')):
        if groups[a] & groups[b]:
            raise ValueError(f'Condition split leakage: {a} and {b}')
    if groups['train'] | groups['validation'] | groups['test'] != universe:
        raise ValueError('Missing or extra conditions in split')
    if 'control' not in groups['train'] or 'control' in groups['test'] | groups['validation']:
        raise ValueError('Control must be training-only')
    return {key: len(items) for key, items in groups.items()}


def write_reproducibility_manifest(output_dir, *, gene_names, perturbations,
                                   split, graph, control_stats, settings):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    info = {
        'gene_count': len(gene_names),
        'gene_order_sha256': array_hash(np.asarray(gene_names).astype('U')),
        'perturbation_vocab': list(perturbations),
        'split': split,
        'graph_sha256': array_hash(graph),
        'control_stats_sha256': array_hash(control_stats),
        'configuration': settings,
    }
    (path/'reproducibility.json').write_text(json.dumps(info, indent=2))
    return info
