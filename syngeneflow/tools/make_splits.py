
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ..data import read_expression, create_split
from ..benchmarks.reproducibility import split_audit


def make_fold_manifests(conditions, *, mode='additive', n_folds=5,
                        seed=42, heldout_count=12):
    if n_folds<1:
        raise ValueError('At least one fold is required')
    folds=[]
    for fold in range(n_folds):
        split=create_split(conditions,mode=mode,seed=seed,fold=fold,
                           holdout_count=heldout_count)
        split_audit(split.as_dict(), set(conditions))
        folds.append(split.as_dict())
    return folds


def main(argv=None):
    p=argparse.ArgumentParser(description='Export explicit SynGeneFlow condition folds')
    p.add_argument('--input',required=True)
    p.add_argument('--condition-key',default='condition')
    p.add_argument('--mode',choices=['additive','unseen','drug'],default='additive')
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--heldout-count',type=int,default=12)
    p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    _,conditions,_=read_expression(args.input,args.condition_key)
    folds=make_fold_manifests(conditions,mode=args.mode,n_folds=args.folds,
                              seed=args.seed,heldout_count=args.heldout_count)
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(folds,indent=2))
    print(f'Saved {len(folds)} condition folds to {destination}')


if __name__=='__main__':
    main()
