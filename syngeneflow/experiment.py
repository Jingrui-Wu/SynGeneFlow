




from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

ABLATIONS = ('none','additive_prior','residual_flow','single_consistency',
             'endpoint_mmd','synergy_mmd','deformation','gene_mask',
             'differential_attention')


def experiment_grid(folds=(0, 1, 2, 3, 4), ablations=('none',)):
    for fold, variant in itertools.product(folds, ablations):
        if variant not in ABLATIONS:
            raise ValueError(f'Unsupported ablation: {variant}')
        yield fold, variant


def generate_commands(base_args, output_dir, folds, ablations):
    result = []
    for fold, variant in experiment_grid(folds, ablations):
        output = Path(output_dir) / f'fold_{fold}' / variant
        command = [sys.executable, '-m', 'syngeneflow.train',
                   *base_args, '--fold', str(fold), '--output', str(output)]
        if variant != 'none':
            command.extend(['--ablate', variant])
        result.append({'fold': fold, 'ablation': variant, 'command': command,
                       'output': str(output)})
    return result


def arguments(argv=None):
    p = argparse.ArgumentParser(description='SynGeneFlow multi-fold experiments')
    p.add_argument('--output-root', default='runs/experiments')
    p.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--ablations', nargs='+', choices=ABLATIONS, default=['none'])
    p.add_argument('--execute', action='store_true')
    p.add_argument('train_args', nargs=argparse.REMAINDER,
                   help='Arguments for syngeneflow.train (after --)')
    return p.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    base = args.train_args[1:] if args.train_args[:1] == ['--'] else args.train_args
    commands = generate_commands(base, args.output_root, args.folds, args.ablations)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output/'experiment_grid.json').write_text(json.dumps(commands, indent=2))
    for run in commands:
        print(' '.join(run['command']), flush=True)
        if args.execute:
            Path(run['output']).mkdir(exist_ok=True, parents=True)
            with (Path(run['output'])/'console.log').open('w') as log:
                subprocess.run(run['command'], check=True, stdout=log, stderr=subprocess.STDOUT)


if __name__ == '__main__':
    main()
