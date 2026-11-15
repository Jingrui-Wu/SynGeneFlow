import numpy as np
import torch
from syngeneflow.data import canonical_condition, components, create_split, SingleCellDataset
from syngeneflow.graph import topk_coexpression_neighbors
from syngeneflow.train import run
from scripts.make_synthetic import make_file


def test_canonical_condition():
    assert canonical_condition("B+A") == "A+B"
    assert canonical_condition("ctrl+A") == "A"
    assert canonical_condition("control+control") == "control"
    assert components("A+B") == ("A", "B")


def test_split_conditions_independent():
    conditions = ["control", "A", "B", "C", "D", "A+B", "A+C", "B+C", "C+D", "A+D"]
    sp = create_split(conditions, mode="additive", seed=17)
    assert "control" in sp.train
    assert set(sp.test).isdisjoint(sp.train)
    assert set(sp.validation).isdisjoint(sp.test)
    assert {"A", "B", "C", "D"}.issubset(sp.train)
    sh = create_split(conditions, mode="unseen", seed=13, holdout_count=1)
    assert all(not any(p in sh.heldout_components for p in components(c))
               for c in sh.train)


def test_split_uses_training_statistics_only():
    conditions = np.array(["control"] * 8 + ["A"] * 8 + ["B"] * 8 + ["A+B"] * 8)
    X = np.ones((32, 8), dtype=np.float32)
    X[:16, 2] = np.arange(16) * .1
    X[24:, 5] = np.arange(8) * 20
    split = create_split(conditions, mode="additive", seed=4)
    ds = SingleCellDataset(X, conditions, ["G" + str(i) for i in range(8)], split,
                           n_genes=2, seed=4, include_perturbed_genes=False)
    assert len(ds.gene_names) == 2
    assert "G5" not in ds.gene_names  
    item = ds.sample(4, torch.arange(len(ds.gene_names)))
    assert all(v.shape[0] == 4 for v in item)


def test_end_to_end_cli(tmp_path):
    file = tmp_path / "synthetic.npz"
    make_file(file, n_genes=12, n_cells_per_condition=8)
    output = tmp_path / "output"
    common = ["--input", str(file), "--output", str(output), "--n-genes", "12",
              "--gene-subset", "8", "--graph-topk", "3", "--hidden", "24",
              "--heads", "4", "--velocity-layers", "1", "--gene-layers", "1",
              "--batch-size", "4", "--graph-fit-cells", "20", "--eval-cells", "3",
              "--ode-steps", "2", "--train-steps", "2", "--save-every", "1",
              "--eval-every", "0", "--device", "cpu"]
    run(common)
    assert (output / "checkpoint_final.pt").exists()
    assert (output / "condition_split.json").exists()
    assert list(output.glob("test_proxies_step_*.csv"))
    run(common + ["--eval-only", "--resume", str(output / "checkpoint_final.pt")])
