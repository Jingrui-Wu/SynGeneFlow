




from __future__ import annotations
from pathlib import Path
import json
import pickle
import numpy as np

DRUG_COMBINATION_TEST = (
    'Panobinostat+Crizotinib',
    'Panobinostat+Curcumin',
    'Panobinostat+SRT1720',
    'Panobinostat+Sorafenib',
    'SRT2104+Alvespimycin',
    'control+Alvespimycin',
    'control+Dacinostat',
)


def canonical(value: str) -> str:
    parts = []
    for item in str(value).split('+'):
        item = item.strip()
        if item.lower() in ('control', 'ctrl', 'vehicle', 'dmso', ''):
            continue
        parts.append(item)
    return '+'.join(sorted(parts)) if parts else 'control'


def active_components(condition: str) -> tuple[str, ...]:
    condition = canonical(condition)
    return tuple(condition.split('+')) if condition != 'control' else ()


def condition_split(conditions, *, mode='additive', seed=42, fold=0,
                    holdout_count=12, test_fraction=0.3,
                    validation_fraction=0.1, curated_drug=True):







    unique = sorted({canonical(c) for c in conditions})
    if 'control' not in unique:
        raise ValueError('Control cells are required')
    active = [c for c in unique if c != 'control']
    rng = np.random.default_rng(seed + fold)
    singles = [c for c in active if len(active_components(c)) == 1]
    combos = [c for c in active if len(active_components(c)) > 1]
    held = []
    if mode == 'additive':
        if not combos:
            raise ValueError('additive split requires multi-perturbation conditions')
        shuffled = rng.permutation(combos).tolist()
        n = min(len(shuffled), max(1, round(test_fraction * len(combos))))
        test = shuffled[:n]
    elif mode == 'unseen':
        candidates = sorted({gene for c in active for gene in active_components(c)})
        if len(candidates) < 2:
            raise ValueError('unseen split needs two or more distinct perturbations')
        held = sorted(rng.choice(candidates, size=min(holdout_count, len(candidates) - 1),
                                 replace=False).tolist())
        test = [c for c in active if set(active_components(c)) & set(held)]
    elif mode == 'drug':
        preferred = [canonical(c) for c in DRUG_COMBINATION_TEST]
        test = sorted(set(preferred) & set(active)) if curated_drug else []
        if not test:
            test = []
            for group in (singles, combos):
                if group:
                    shuffled = rng.permutation(group).tolist()
                    test.extend(shuffled[:max(1, round(len(group) * test_fraction))])
        if not test:
            raise ValueError('Drug split requires perturbation conditions')
    else:
        raise ValueError('mode must be additive, unseen, or drug')
    remaining = sorted(set(active) - set(test))
    
    candidate = [c for c in remaining if len(active_components(c)) > 1]
    if not candidate:
        candidate = list(remaining)
    rng.shuffle(candidate)
    n_val = min(len(candidate), round(validation_fraction * len(candidate)))
    validation = sorted(candidate[:n_val])
    train = ['control'] + sorted(set(remaining) - set(validation))
    if not (set(train) - {'control'}):
        raise ValueError('No training perturbations left after splitting')
    return {'train': train, 'validation': validation, 'test': sorted(test),
            'heldout_components': held}


def read_split_manifest(path: str | Path, *, fold: int = 0):





    path = Path(path)
    if path.suffix == '.json':
        obj = json.loads(path.read_text())
    elif path.suffix in ('.pkl', '.pickle'):
        with path.open('rb') as handle:
            obj = pickle.load(handle)
    else:
        raise ValueError('Expected .json, .pkl, or .pickle split manifest')
    if isinstance(obj, list):
        if not 0 <= fold < len(obj):
            raise IndexError(f'fold {fold} is outside available folds (n={len(obj)})')
        obj = obj[fold]
    if not isinstance(obj, dict):
        raise TypeError('Split manifest must contain a dictionary or list thereof')
    return obj


def resolve_manifest(conditions, manifest):
    unique = {canonical(c) for c in conditions}
    test = {canonical(c) for c in manifest.get('test', [])}
    val = {canonical(c) for c in manifest.get('validation', [])}
    if 'control' in test | val or test & val:
        raise ValueError('Overlapping validation/test conditions or held-out control')
    unknown = (test | val) - unique
    if unknown:
        raise ValueError(f'Unknown test/validation conditions: {sorted(unknown)[:10]}')
    train = unique - test - val
    if 'train' in manifest:
        specified = {canonical(c) for c in manifest['train']}
        
        
        acceptable_full = specified == train or specified | {'control'} == train
        listed_multi = {c for c in specified if len(active_components(c)) > 1}
        expected_multi = {c for c in train if len(active_components(c)) > 1}
        if not acceptable_full and listed_multi != expected_multi:
            raise ValueError('Explicit training conditions conflict with manifest test/validation sets')
    return {'train': ['control'] + sorted(train - {'control'}),
            'validation': sorted(val), 'test': sorted(test),
            'heldout_components': list(manifest.get('heldout_components', []))}
