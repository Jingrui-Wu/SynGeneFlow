
import argparse
from pathlib import Path
import numpy as np


def make_file(path, seed=19, n_genes=32, n_cells_per_condition=18):
    rng = np.random.default_rng(seed)
    names = np.array(["A", "B", "C", *[f"G{i}" for i in range(n_genes-3)]])
    conditions = ["control", "A+control", "B+control", "C+control",
                  "A+B", "A+C", "B+C"]
    x, labels = [], []
    for condition in conditions:
        parts = [p for p in condition.split("+") if p != "control"]
        base = rng.uniform(0.4, 1.7, (n_cells_per_condition, n_genes)).astype(np.float32)
        shift = sum((rng.normal(0.1, .05, (1, n_genes)) for _ in parts),
                    np.zeros((1, n_genes)))
        synergy = 0.2 if len(parts) > 1 else 0.0
        x.append(np.maximum(0, base + shift + synergy).astype(np.float32))
        labels.extend([condition] * n_cells_per_condition)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=np.concatenate(x), conditions=np.array(labels), gene_names=names)
    print(f"Wrote synthetic file {path}; it is NOT biological data")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/synthetic_demo.npz")
    a = p.parse_args()
    make_file(a.output)
