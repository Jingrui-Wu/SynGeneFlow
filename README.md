# SynGeneFlow

Synergy-aware residual flow matching for single-cell perturbation prediction.

**Paper:** [Synergy-Aware Residual Flow Matching for Single-Cell Perturbation Prediction](https://arxiv.org/abs/2610.25021) — IEEE BIBM 2026.

## Installation

Python 3.10 or later is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,test]'
```

The core dependencies are PyTorch, NumPy, and SciPy. The `data` extra enables AnnData (`.h5ad`) input, and the `test` extra installs pytest. Choose a PyTorch build appropriate for your CPU or CUDA environment.

## Synthetic example

Generate a small artificial dataset and run a short CPU training example:

```bash
python scripts/make_synthetic.py --output data/synthetic_demo.npz
python -m syngeneflow.train \
  --input data/synthetic_demo.npz \
  --output runs/synthetic_demo \
  --n-genes 32 --gene-subset 10 --graph-topk 3 \
  --hidden 16 --heads 4 --gene-layers 1 --velocity-layers 1 \
  --batch-size 4 --graph-fit-cells 16 --eval-cells 2 \
  --train-steps 2 --ode-steps 2 --eval-every 0 --save-every 2 \
  --device cpu
```

This example uses synthetic data to exercise the pipeline; it does not reproduce the paper's biological experiments.

## Dataset configurations

The `configs/` directory contains configurations for Norman additive, Norman held-out, and combosciplex experiments. Supply your own preprocessed dataset:

```bash
python -m syngeneflow.tools.run_config \
  --config configs/norman_additive.json \
  --input /path/to/norman.h5ad \
  --output runs/norman_additive \
  --dry-run
```

Remove `--dry-run` to start training. Confirm that condition labels, preprocessing, gene order, and split definitions match the intended experiment. Reference gene-order files are included in `resources/`; biological datasets and trained checkpoints are not included.

## Tools and tests

```bash
python -m syngeneflow.tools.inspect_data --help
python -m syngeneflow.tools.make_splits --help
python -m syngeneflow.tools.export_predictions --help
python -m syngeneflow.benchmarks.assess --help
python -m pytest -q
```

## Repository contents

- `syngeneflow/`: model, training, inference, transport, and evaluation code.
- `configs/`: experiment configurations.
- `resources/`: reference gene orders.
- `scripts/`: synthetic data example.
- `tests/`: unit and integration tests.

## License

MIT license. See [LICENSE](LICENSE).
