
import numpy as np
import torch
from pathlib import Path
from scripts.make_synthetic import make_file
from syngeneflow.tools.inspect_data import inspect_dataset
from syngeneflow.tools.make_splits import make_fold_manifests
from syngeneflow.tools.export_predictions import generate_populations
from syngeneflow.train import run
from syngeneflow.inference import load_inference_bundle
from syngeneflow.data import read_expression


def test_inspect_file_and_multi_fold_manifest(tmp_path):
    archive=tmp_path/'mini.npz'
    make_file(archive,n_genes=9,n_cells_per_condition=7)
    info=inspect_dataset(archive)
    assert info['expression']['n_cells']>20
    assert info['n_single_conditions']>=2
    X,conditions,genes=read_expression(archive,'condition')
    splits=make_fold_manifests(conditions,n_folds=3,mode='additive')
    assert len(splits)==3
    for split in splits:
        assert 'control' in split['train']
        assert set(split['train']).isdisjoint(split['test'])


def test_generate_condition_populations_from_saved_checkpoint(tmp_path):
    archive=tmp_path/'mini.npz'
    make_file(archive,n_genes=10,n_cells_per_condition=7)
    folder=tmp_path/'runs'
    run(['--input',str(archive),'--output',str(folder),'--n-genes','10',
         '--gene-subset','8','--graph-topk','2','--hidden','16',
         '--heads','4','--gene-layers','1','--velocity-layers','1',
         '--batch-size','4','--graph-fit-cells','16','--eval-cells','2',
         '--train-steps','1','--ode-steps','1','--eval-every','0','--save-every','0',
         '--device','cpu'])
    model,ckpt=load_inference_bundle(folder/'checkpoint_final.pt')
    data=np.load(archive)
    x=data['X'][np.array(data['conditions'])=='control']
    lookup={g:i for i,g in enumerate(data['gene_names'])}
    controls=x[:,[lookup[g] for g in ckpt['gene_names']]]
    labels=['A+B','B+C']
    generated,tagged=generate_populations(model,ckpt,controls,labels,
                                         cells_per_condition=3,batch_size=2,
                                         steps=1,core_size=4,max_context=8)
    assert generated.shape==(6,len(ckpt['gene_names']))
    assert tagged.tolist()==['A+B']*3+['B+C']*3
    assert np.isfinite(generated).all()
