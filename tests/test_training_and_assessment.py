
import numpy as np
import pytest
import torch
from pathlib import Path
from syngeneflow.training.ema import ExponentialMovingAverage
from syngeneflow.training.checkpoint import atomic_torch_save, checkpoint_summary
from syngeneflow.benchmarks.bootstrap import condition_bootstrap, paired_condition_comparison
from syngeneflow.benchmarks.assess import evaluate_populations,aggregate_conditions,load_population
from syngeneflow.train import run
from syngeneflow.inference import predict_from_checkpoint
from scripts.make_synthetic import make_file


def test_ema_restores_live_parameters():
    layer=torch.nn.Linear(3,4)
    ema=ExponentialMovingAverage(layer,.9)
    with torch.no_grad():
        layer.weight.add_(1.)
    ema.update(layer)
    original=layer.weight.detach().clone()
    with ema.average_parameters(layer):
        assert torch.isfinite(layer.weight).all()
    assert torch.equal(layer.weight,original)
    second=ExponentialMovingAverage(layer,.9)
    second.load_state_dict(ema.state_dict())
    assert second.updates==1


def test_atomic_checkpoint_roundtrip(tmp_path):
    path=tmp_path/'check.pt'
    obj={'step':10,'model':{'w':torch.ones(2)},'config':{},
         'gene_names':['A'],'vocab':['control'],'split':{'train':['control']},
         'graph':torch.ones(1,1)}
    atomic_torch_save(obj,path)
    assert checkpoint_summary(torch.load(path,weights_only=True))['step']==10
    assert not list(tmp_path.glob('.partial-*'))


def test_bootstrap_conditions():
    ci=condition_bootstrap(np.arange(10),seed=0,replications=50)
    assert ci.n_conditions==10
    assert ci.lower<=ci.estimate<=ci.upper
    gain=paired_condition_comparison([2,3,4],[1,2,3],replications=30)
    assert gain.estimate==1.0
    assert np.isnan(condition_bootstrap([2.],replications=10).lower)


def test_independent_population_assessment():
    ctrl=np.ones((8,8),np.float32)
    a=ctrl+np.arange(8)[None]*0.1
    b=ctrl+np.arange(8)[None]*0.03
    double=ctrl+np.arange(8)[None]*0.17
    real=np.concatenate([ctrl,a,b,double],0)
    labels=['control']*8+['A']*8+['B']*8+['A+B']*8
    pred=double.copy()
    results=evaluate_populations(pred,['A+B']*8,real,labels,[f'G{i}' for i in range(8)],
                                 train_conditions=['control','A','B'],max_cells=8)
    assert len(results)==1
    assert results[0]['mean_MSE']==0.0
    assert 'synergy_Pearson' in results[0]
    summary=aggregate_conditions(results,bootstrap_reps=20)
    assert summary['mean_MSE']['estimate']==0.0


def test_population_assessment_avoids_test_single_reference():
    ctrl=np.ones((8,8),np.float32)
    obs=np.concatenate([ctrl,ctrl+1,ctrl+2,ctrl+3],0)
    labels=['control']*8+['A']*8+['B']*8+['A+B']*8
    scored=evaluate_populations(ctrl+3,['A+B']*8,obs,labels,[f'G{i}' for i in range(8)],
                                train_conditions=['control','A'],max_cells=8)
    assert 'synergy_MSE' not in scored[0]


def test_ema_accumulation_full_gene_inference(tmp_path):
    file=tmp_path/'input.npz'
    make_file(file,n_genes=14,n_cells_per_condition=6)
    folder=tmp_path/'runs'
    train_args=['--input',str(file),'--output',str(folder),'--n-genes','14',
                '--gene-subset','10','--graph-topk','3','--hidden','16',
                '--heads','4','--gene-layers','1','--velocity-layers','1',
                '--batch-size','4','--graph-fit-cells','16','--eval-cells','2',
                '--train-steps','2','--grad-accumulation','2','--ema-decay','.9',
                '--eval-ema','--eval-full-genes','--gene-chunk-core','4',
                '--gene-chunk-context','10','--ode-steps','2','--eval-every','0',
                '--save-every','2','--device','cpu']
    run(train_args)
    checkpoint=folder/'checkpoint_final.pt'
    ckpt=torch.load(checkpoint,weights_only=False)
    assert ckpt['ema']['updates']==2
    archive=np.load(file)
    x=archive['X'][:3]
    names=archive['gene_names'].tolist()
    estimate, ordered=predict_from_checkpoint(checkpoint,x,names,'A+B',
                                              n_samples=2,ode_steps=2,batch_size=2,
                                              max_context=10,core_size=4)
    assert estimate.shape==(2,len(ordered))
    assert np.isfinite(estimate).all()


def test_offline_assessment_aligns_raw_observations_to_prediction_genes(tmp_path):

    import json
    from syngeneflow.benchmarks.assess import main
    raw=tmp_path/'all.npz'
    rng=np.random.default_rng(11)
    X=rng.normal(size=(24,12)).astype(np.float32)
    labels=np.array(['control']*12+['A+B']*12)
    names=np.array([f'G{i}' for i in range(12)])
    np.savez(raw,X=X,conditions=labels,gene_names=names)
    gen=tmp_path/'pred.npz'
    np.savez(gen,X=X[12:,:8],conditions=np.array(['A+B']*12),gene_names=names[:8])
    out=tmp_path/'assessment'
    main(['--predictions',str(gen),'--observations',str(raw),
          '--output',str(out),'--bootstrap','20'])
    report=json.loads((out/'summary.json').read_text())
    assert report['n_conditions']==1
    assert report['metrics']['mean_MSE']['estimate']>=0
    assert report['metrics']['mean_MSE']['lower'] is None
