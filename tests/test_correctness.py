import itertools
import math
import numpy as np
import pytest
import torch
from repro.graphs import GCN,EdgeConv,hierarchical_edges,temporal_edges,HierarchicalGraph
from repro.metrics import ctc_decode,corpus_wer,translation_scores
from repro.data import collate,build_vocab,encode_targets,safe_path
from repro.runtime import batch_plan
from repro.train import GlobalBatchSampler

torch.set_num_threads(2)

def test_hsg_edges_bidirectional_rectangular():
    edges=hierarchical_edges(4,6,2,3)
    forward=edges[:,:24]
    assert torch.equal(edges[:,24:],forward.flip(0))
    assert forward[1,5].item()==26
    assert forward[1,6].item()==24
    assert forward[1,23].item()==29
    with pytest.raises(ValueError): hierarchical_edges(5,6,2,3)

def test_gcn_matches_dense_reference_and_backward():
    edges=hierarchical_edges(2,2,1,1)
    x=torch.randn(3,5,4,requires_grad=True)
    layer=GCN(4)
    actual=layer(x,edges)
    adjacency=torch.eye(5)
    adjacency[edges[1],edges[0]]=1
    deg=adjacency.sum(1).rsqrt()
    normalized=deg[:,None]*adjacency*deg[None,:]
    expected=normalized @ layer.lin(x)+layer.bias
    torch.testing.assert_close(actual,expected)
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_temporal_nearest_pairs_no_cross_frame_edges():
    x=torch.tensor([[[0.],[10.]],[[1.],[30.]],[[2.],[50.]]])
    edge=temporal_edges(x,1)
    assert edge.tolist()==[[0,2,2,4],[2,4,0,2]]
    assert torch.all((edge[0]//2-edge[1]//2).abs()==1)

def test_hsg_gradient_both_inputs():
    high=torch.randn(2,4,4,4,requires_grad=True)
    low=torch.randn(2,8,2,2,requires_grad=True)
    result=HierarchicalGraph(4,8)(high,low)
    result.square().mean().backward()
    assert high.grad.abs().sum()>0 and low.grad.abs().sum()>0

def test_edgeconv_matches_neighbor_loop_and_isolated_nodes():
    torch.manual_seed(11)
    layer=EdgeConv(4)
    x=torch.randn(2,4,4,requires_grad=True)
    edges=torch.tensor([[0,1,2],[2,2,0]])
    result=layer(x,edges)
    expected=[]
    for graph in x:
        values=[]
        for destination in range(4):
            sources=edges[0,edges[1]==destination]
            values.append(layer.mlp(torch.cat([graph[destination].expand(len(sources),-1),
                                               graph[sources]-graph[destination]],-1)).max(0).values
                          if len(sources) else graph.new_zeros(4))
        expected.append(torch.stack(values))
    torch.testing.assert_close(result,torch.stack(expected))
    result.sum().backward()
    assert torch.isfinite(x.grad).all()

def test_ctc_beam_matches_exhaustive_paths():
    probs=np.array([[.1,.6,.3],[.5,.4,.1],[.2,.2,.6]])
    totals={}
    for path in itertools.product(range(3),repeat=3):
        collapsed=tuple(t for t,_ in itertools.groupby(path) if t)
        totals[collapsed]=totals.get(collapsed,0)+math.prod(probs[i,t] for i,t in enumerate(path))
    assert ctc_decode(np.log(probs),100)==list(max(totals,key=totals.get))
    assert ctc_decode(np.log([[.01,.99],[.99,.01],[.01,.99]]),1)==[1,1]

def test_wer_counts():
    score=corpus_wer(['a b c','d'],['a x c y',''])
    assert score['wer']==75 and score['sub']==25 and score['ins']==25 and score['del']==25

def test_translation_metrics():
    score=translation_scores(['a b c d e'],['a b c d e'])
    assert score['bleu4']==pytest.approx(100)
    assert score['rouge']==pytest.approx(100,abs=1e-5)

@pytest.mark.parametrize('available,expected',[ (1,1),(2,2),(3,3),(4,3),(8,6)])
def test_gpu_plan(available,expected):
    plan=batch_plan(available,6,1)
    assert plan['world_size']==expected
    assert plan['accumulation']*plan['world_size']*plan['micro_batch']==6

def test_global_sampler_no_overlap():
    shards=[list(GlobalBatchSampler(17,6,r,3,0,0)) for r in range(3)]
    assert all(len(s)==4 for s in shards)
    assert len(set(itertools.chain.from_iterable(shards)))==12
    with pytest.raises(ValueError): batch_plan(4,6,1,4)

def test_collation_ctc_repeats_and_vocab():
    batch=collate([(torch.ones(9,4),{'id':'a'}),(torch.ones(4,4),{'id':'b'})])
    assert batch['lengths'].tolist()==[24,16]
    assert batch['video'].shape==(2,24,4)
    vocab=build_vocab([{'gloss':'a b'}],'gloss')
    _,lengths,required=encode_targets([{'gloss':'a a b'}],vocab,'gloss')
    assert lengths.item()==3 and required.item()==4

def test_path_escape(tmp_path):
    with pytest.raises(ValueError): safe_path(tmp_path,'../secret')
