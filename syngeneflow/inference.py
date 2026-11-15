




from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from .data import read_expression, canonical_condition, components, dense_rows
from .graph import induced_graph_mask
from .model import SynGeneFlow
from .chunking import generate_chunked, build_gene_chunks


def load_inference_bundle(checkpoint, device='cpu', use_ema=None):
    
    bundle = torch.load(checkpoint, map_location='cpu', weights_only=False)
    settings = bundle['config']
    gene_names = list(bundle['gene_names'])
    vocabulary = list(bundle['vocab'])
    mapping = {p: i for i, p in enumerate(vocabulary)}
    linking = {g: i for i, g in enumerate(gene_names)}
    gene_id = [-1] * len(vocabulary)
    for p, index in mapping.items():
        gene_id[index] = linking.get(p, -1)
    model = SynGeneFlow(len(gene_names), len(vocabulary), control_id=mapping['control'],
                        perturb_to_gene=gene_id, hidden=settings['hidden'],
                        heads=settings['heads'], gene_layers=settings['gene_layers'],
                        velocity_layers=settings['velocity_layers'],
                        edge_hidden=settings['edge_hidden'])
    if use_ema is None:
        use_ema = bool(settings.get('eval_ema', False))
    if use_ema and bundle.get('ema') is None:
        raise ValueError('EMA was requested, but checkpoint lacks EMA weights')
    weights = bundle['ema']['shadow'] if use_ema else bundle['model']
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return model, bundle


def align_controls(expression, genes, checkpoint_genes):
    lookup = {str(v): i for i, v in enumerate(genes)}
    absent = sorted(set(checkpoint_genes) - set(lookup))
    if absent:
        raise ValueError(f'Control dataset is missing required genes: {absent[:10]}')
    selected = np.asarray([lookup[g] for g in checkpoint_genes], dtype=np.int64)
    return dense_rows(expression, np.arange(expression.shape[0]), selected)


def perturbation_ids(condition, vocab, batch_size):
    lookup = {p: i for i, p in enumerate(vocab)}
    terms = list(components(canonical_condition(condition)))
    unknown = [p for p in terms if p not in lookup]
    if unknown:
        raise ValueError(f'Unknown perturbations {unknown}: cannot infer untrained ID embeddings')
    ids = [lookup[p] for p in terms]
    return torch.tensor(ids + [lookup['control']] * max(1, 2-len(ids)),
                        dtype=torch.long)[None].expand(batch_size, -1)


@torch.inference_mode()
def predict_from_checkpoint(checkpoint, control_expression, genes, condition,
                            *, n_samples=None, ode_steps=20, tau=None, device='cpu',
                            integrator='euler', batch_size=32,
                            seed=42, clamp_output=False, core_size=96, max_context=None,
                            use_ema=None):
    model, ckpt = load_inference_bundle(checkpoint, device=device, use_ema=use_ema)
    ordered_genes = ckpt['gene_names']
    source = align_controls(control_expression, genes, ordered_genes)
    if n_samples is not None:
        rng = np.random.default_rng(seed)
        source = source[rng.choice(len(source), n_samples, replace=True)]
    neighbors = ckpt['graph']
    if max_context is None:
        max_context = max(1, min(len(ordered_genes), ckpt['config'].get('gene_subset', 256)))
    core_size = min(core_size, max_context)
    chunks = build_gene_chunks(neighbors, core_size=core_size, max_context=max_context)
    control_std = torch.as_tensor(ckpt['control_std'], device=device)
    if tau is None:
        tau = ckpt['config']['tau']
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = []
    for offset in range(0, len(source), batch_size):
        batch = torch.as_tensor(source[offset:offset+batch_size], dtype=torch.float32, device=device)
        condition_ids = perturbation_ids(condition, ckpt['vocab'], len(batch)).to(device)
        result = generate_chunked(
            model, batch, condition_ids, neighbors, control_std, chunks=chunks,
            steps=ode_steps, tau=tau, integrator=integrator,
            clamp_output=clamp_output, generator=generator)
        generated.append(result.cpu().numpy())
    return np.concatenate(generated, axis=0), ordered_genes


def arguments(argv=None):
    parser = argparse.ArgumentParser(description='SynGeneFlow inference')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='Control .npz or .h5ad')
    parser.add_argument('--condition', required=True, help='e.g. A+B')
    parser.add_argument('--condition-key', default='condition')
    parser.add_argument('--layer', default=None)
    parser.add_argument('--output', required=True, help='Output .npz')
    parser.add_argument('--n-samples', type=int, default=None)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--integrator', choices=['euler', 'heun', 'rk4'], default='euler')
    parser.add_argument('--clamp-output', action='store_true')
    parser.add_argument('--raw-weights', action='store_true',
                        help='Override EMA selection from the training configuration')
    parser.add_argument('--core-size', type=int, default=96)
    parser.add_argument('--max-context', type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    X, conditions, genes = read_expression(args.input, args.condition_key, args.layer)
    rows = np.flatnonzero(conditions == 'control')
    if not len(rows):
        raise ValueError('Input file must include at least one control cell')
    expression = X[rows]
    prediction, names = predict_from_checkpoint(
        args.checkpoint, expression, genes, args.condition,
        n_samples=args.n_samples, ode_steps=args.steps, batch_size=args.batch_size,
        seed=args.seed, tau=args.tau, device=args.device,
        integrator=args.integrator, clamp_output=args.clamp_output,
        core_size=args.core_size, max_context=args.max_context,
        use_ema=False if args.raw_weights else None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=prediction, gene_names=np.asarray(names),
                        condition=np.asarray([canonical_condition(args.condition)] * len(prediction)))
    print(f'Saved {len(prediction)} generated cells, {len(names)} genes to {out}')


if __name__ == '__main__':
    main()
