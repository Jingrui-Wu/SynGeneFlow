
import pytest
import torch
from syngeneflow.chunking import build_gene_chunks, generate_chunked


def neighbors(n=19):
    return torch.tensor([[(g+1)%n, (g-1)%n] for g in range(n)],dtype=torch.long)


def test_chunk_plan_covers_every_gene_exactly_once():
    plans = build_gene_chunks(neighbors(),core_size=5,max_context=9)
    written = torch.cat([p.core for p in plans]).sort().values
    assert torch.equal(written,torch.arange(19))
    for plan in plans:
        assert len(plan.selected) <= 9
        assert torch.equal(plan.selected[plan.core_positions], plan.core)


def test_chunk_plan_with_permutation():
    plans=build_gene_chunks(neighbors(),core_size=7,max_context=12,order=list(range(18,-1,-1)))
    assert sorted(torch.cat([p.core for p in plans]).tolist())==list(range(19))


def test_chunk_plan_rejects_invalid_context():
    with pytest.raises(ValueError): build_gene_chunks(neighbors(),core_size=5,max_context=3)


class IdentityModel:
    def generate(self,control,pert,indices,mask,std,**kwargs):
        return control + indices[None].to(control.device) * 0.1


def test_chunk_generation_reconstructs_full_matrix():
    x = torch.randn(3,19)
    out=generate_chunked(IdentityModel(),x,torch.zeros(3,2,dtype=torch.long),
                         neighbors(),torch.ones(19),core_size=4,max_context=7)
    assert torch.allclose(out,x+0.1*torch.arange(19)[None],atol=1e-6)
