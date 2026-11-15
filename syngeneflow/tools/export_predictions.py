





from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..data import read_expression, log1p_normalize
from ..benchmarks.splits import canonical
from ..inference import load_inference_bundle, align_controls, perturbation_ids
from ..chunking import build_gene_chunks, generate_chunked


def generate_populations(model, checkpoint_metadata, control_matrix, conditions,
                         *, cells_per_condition=128, batch_size=32, steps=20,
                         core_size=96, max_context=None, tau=None,
                         integrator='euler', seed=42, device='cpu'):





    if cells_per_condition < 1 or batch_size < 1:
        raise ValueError('cells_per_condition and batch_size must be positive')
    if len(control_matrix) < 1:
        raise ValueError('At least one control cell is required')
    if len(set(conditions)) != len(conditions):
        raise ValueError('Duplicate requested perturbation conditions')
    control_matrix=np.asarray(control_matrix,dtype=np.float32)
    n_genes=len(checkpoint_metadata['gene_names'])
    if control_matrix.ndim != 2 or control_matrix.shape[1] != n_genes:
        raise ValueError('Control matrix must match checkpoint gene order')
    neighbors=checkpoint_metadata['graph']
    std=torch.as_tensor(checkpoint_metadata['control_std'], device=device)
    if max_context is None:
        max_context=min(n_genes, checkpoint_metadata['config'].get('gene_subset',256))
    chunks=build_gene_chunks(neighbors,core_size=min(core_size,max_context),
                             max_context=max_context)
    if tau is None:
        tau=checkpoint_metadata['config']['tau']
    rng=np.random.default_rng(seed)
    generator=torch.Generator(device=device).manual_seed(seed)
    predictions=[]
    labels=[]
    for condition in conditions:
        label=canonical(condition)
        indices=rng.integers(0,len(control_matrix),size=cells_per_condition)
        for offset in range(0,cells_per_condition,batch_size):
            rows=indices[offset:offset+batch_size]
            ctrl=torch.as_tensor(control_matrix[rows],dtype=torch.float32,device=device)
            condition_ids=perturbation_ids(label,checkpoint_metadata['vocab'],len(rows)).to(device)
            with torch.inference_mode():
                output=generate_chunked(model,ctrl,condition_ids,neighbors,std,
                                        chunks=chunks,steps=steps,tau=tau,
                                        generator=generator,integrator=integrator)
            predictions.append(output.cpu().numpy())
            labels.extend([label]*len(rows))
    if not predictions:
        raise ValueError('No conditions to generate; check the checkpoint split')
    return np.concatenate(predictions,0),np.asarray(labels,dtype=str)


def arguments(argv=None):
    p=argparse.ArgumentParser(description='Export SynGeneFlow held-out perturbation predictions')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',required=True,help='Control input .npz/.h5ad')
    p.add_argument('--output',required=True,help='Output generated .npz')
    p.add_argument('--split',choices=['train','validation','test'],default='test')
    p.add_argument('--conditions',nargs='+',default=None,
                   help='Override split by providing explicit perturbation names')
    p.add_argument('--condition-key',default='condition')
    p.add_argument('--layer',default=None)
    p.add_argument('--transform',choices=['auto','as_is','counts'],default='auto')
    p.add_argument('--cells-per-condition',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--core-size',type=int,default=96)
    p.add_argument('--max-context',type=int,default=None)
    p.add_argument('--tau',type=float,default=None)
    p.add_argument('--integrator',choices=['euler','heun','rk4'],default='euler')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    p.add_argument('--raw-weights',action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args=arguments(argv)
    model,metadata=load_inference_bundle(args.checkpoint,device=args.device,
                                         use_ema=False if args.raw_weights else None)
    X,labels,genes=read_expression(args.input,args.condition_key,args.layer)
    transform=metadata['config']['transform'] if args.transform=='auto' else args.transform
    if transform=='counts':
        X=log1p_normalize(X)
    controls=align_controls(X[np.flatnonzero(labels=='control')],genes,
                            metadata['gene_names'])
    conditions=args.conditions if args.conditions else [c for c in metadata['split'][args.split]
                                                        if c!='control']
    expression,generated_labels=generate_populations(
        model,metadata,controls,conditions,cells_per_condition=args.cells_per_condition,
        batch_size=args.batch_size,steps=args.steps,tau=args.tau,
        core_size=args.core_size,max_context=args.max_context,
        integrator=args.integrator,seed=args.seed,device=args.device)
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,X=expression,conditions=generated_labels,
                        gene_names=np.asarray(metadata['gene_names']))
    info={'checkpoint':str(args.checkpoint),'source':str(args.input),
          'transform':transform,'generated_labels':list(conditions),
          'cells_per_condition':args.cells_per_condition,'integrator':args.integrator,
          'ode_steps':args.steps,'seed':args.seed,
          'shape':list(expression.shape),'chunk_context':args.max_context}
    dest.with_suffix('.json').write_text(json.dumps(info,indent=2))
    print(f'Saved {len(expression)} generated cells to {dest}')


if __name__=='__main__':
    main()
