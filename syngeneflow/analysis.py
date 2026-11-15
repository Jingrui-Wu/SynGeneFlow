





from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read_table(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def summarize_column(rows, name):
    values = []
    for row in rows:
        try:
            v = float(row[name])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(v):
            values.append(v)
    if not values:
        return {'count': 0, 'mean': None, 'std': None, 'median': None}
    a = np.asarray(values, np.float64)
    return {'count': len(a), 'mean': float(a.mean()),
            'std': float(a.std(ddof=1)) if len(a) > 1 else 0.,
            'median': float(np.median(a))}


def aggregate_folds(root, stem='test_proxies_step_'):
    root = Path(root)
    files = sorted(root.rglob(stem+'*.csv'))
    if not files:
        raise FileNotFoundError(f'No matching evaluation CSVs in {root}')
    
    
    latest = {}
    for path in files:
        key = path.parent
        if key not in latest or path.name > latest[key].name:
            latest[key] = path
    all_rows = []
    for path in latest.values():
        per_condition = read_table(path)
        by_run = {}
        for key in per_condition[0].keys() if per_condition else ():
            if key in ('condition','n_target_cells_in_dataset','n_generated'):
                continue
            statistic = summarize_column(per_condition, key)
            if statistic['mean'] is not None:
                by_run[key] = statistic['mean']
        all_rows.append(by_run)
    metrics = sorted({key for row in all_rows for key in row})
    return {'n_runs': len(all_rows),
            'files': [str(p) for p in latest.values()],
            'metrics': {k: summarize_column(all_rows, k) for k in metrics}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Aggregate SynGeneFlow evaluation CSVs')
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args(argv)
    result = aggregate_folds(args.root)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
