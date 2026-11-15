
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from ..data import read_expression
from ..benchmarks.splits import active_components
from ..preprocessing import describe_expression


def inspect_dataset(input_path, *, condition_key='condition', layer=None):
    expression, conditions, gene_names = read_expression(input_path,condition_key,layer)
    stats = describe_expression(expression)
    counter = Counter(conditions)
    multiplicity = Counter(len(active_components(c)) for c in counter)
    sample_counts = sorted((int(n),c) for c,n in counter.items())
    return {
        'source': str(input_path),
        'expression': stats.as_dict(),
        'control_cells': int(counter.get('control',0)),
        'n_conditions': len(counter),
        'n_single_conditions': multiplicity.get(1,0),
        'n_combination_conditions': sum(n for k,n in multiplicity.items() if k>1),
        'max_combination_size': max(multiplicity) if multiplicity else 0,
        'smallest_condition_groups': [{'condition':c,'cells':n} for n,c in sample_counts[:10]],
        'largest_condition_groups': [{'condition':c,'cells':n} for n,c in sample_counts[-10:]],
        'gene_name_examples': list(map(str,gene_names[:15])),
        'warnings': _warnings(stats,counter),
    }


def _warnings(stats, counter):
    warnings=[]
    if stats.nonfinite_count:
        warnings.append('Expression contains NaN/Inf; resolve before training.')
    if stats.negative_fraction:
        warnings.append('Negative values are present; input may be scaled/residualized, not raw counts.')
    if stats.zero_fraction < .01:
        warnings.append('Nearly no zeros; inspect whether data are normalized or imputed.')
    if counter.get('control',0) < 2:
        warnings.append('Fewer than two control cells; distributional training is unsupported.')
    if any(n < 2 for c,n in counter.items() if c != 'control'):
        warnings.append('At least one perturbation has fewer than two observed cells.')
    return warnings


def main(argv=None):
    p=argparse.ArgumentParser(description='Inspect SynGeneFlow single-cell datasets')
    p.add_argument('--input',required=True)
    p.add_argument('--condition-key',default='condition')
    p.add_argument('--layer',default=None)
    p.add_argument('--output',default=None)
    args=p.parse_args(argv)
    info=inspect_dataset(args.input,condition_key=args.condition_key,layer=args.layer)
    report=json.dumps(info,indent=2)
    if args.output:
        dest=Path(args.output)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(report)
    else:
        print(report)


if __name__=='__main__':
    main()
