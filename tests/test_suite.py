import json
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
from repro.suite import experiments, BASE
from repro.ablation import distances, neighbors, SAGE, GATv2
from repro.model import SignModel
from scripts.run_suite import execute, signature, complete_checkpoint

torch.set_num_threads(2)


def test_inventory_covers_paper_rows_and_one_parameter_k_sweeps():
    rows=experiments()
    assert len(rows)==68 and len({r['id'] for r in rows})==68
    assert sum(r['source']=='Table 1' for r in rows)==12
    assert sum(r['source']=='Table 13a' for r in rows)==8
    for row in rows:
        if row['source']=='Figure 6':
            key=next(iter(row['changes']))
            assert sum(a!=b for a,b in zip(row['changes'][key],BASE[key]))==1
    assert next(r for r in rows if r['id']=='modules_lsg1_lsg2')['changes']['module_mask']==[True,False,False,True,False,False]


def test_distances_and_sage_against_independent_reference():
    x=torch.tensor([[1.,0.],[0.,2.],[-1.,0.]])
    torch.testing.assert_close(distances(x,x,'chebyshev'), (x[:,None]-x[None]).abs().amax(-1))
    torch.testing.assert_close(distances(x,x,'cosine'),torch.tensor([[0.,1.,2.],[1.,0.,1.],[2.,1.,0.]]))
    edges=torch.tensor([[0,2],[1,1]])
    layer=SAGE(2)
    out=layer(x[None],edges)
    mean=torch.zeros_like(x); mean[1]=(x[0]+x[2])/2
    torch.testing.assert_close(out[0],layer.root(x)+layer.neighbor(mean))


def test_gatv2_matches_destination_softmax_reference():
    torch.manual_seed(31)
    graph=GATv2(3); x=torch.randn(2,4,3,requires_grad=True)
    edges=torch.tensor([[0,2,1],[1,1,3]])
    out=graph(x,edges); expected=[]
    for sample in x:
        source=graph.source(sample); target=graph.target(sample); values=[]
        for node in range(4):
            ids=torch.cat([edges[0,edges[1]==node],torch.tensor([node])])
            scores=(torch.nn.functional.leaky_relu(source[ids]+target[node],.2)*graph.attention).sum(-1)
            values.append((scores.softmax(0)[:,None]*source[ids]).sum(0)+graph.bias)
        expected.append(torch.stack(values))
    torch.testing.assert_close(out,torch.stack(expected))
    out.square().sum().backward(); assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('row',experiments(),ids=lambda row:row['id'])
def test_each_experiment_has_finite_backbone_gradients_and_no_unused_trainable_parameters(row):
    torch.manual_seed(5)
    cfg={**BASE,**row['changes'],'hidden_size':16,'allow_random_init':True}
    model=SignModel(cfg,['<blank>','<unk>','a'])
    output=model.backbone(torch.randn(1,3,2,224,224))
    assert output.shape==(2,model.input_dim) and torch.isfinite(output).all()
    output.square().mean().backward()
    unused=[name for name,p in model.backbone.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, unused
    assert all(torch.isfinite(p.grad).all() for p in model.backbone.parameters() if p.grad is not None)


def test_checkpoint_completion_and_portable_identity(tmp_path):
    path=tmp_path/'last.pt'
    torch.save({'epoch':49,'epoch_complete':False},path)
    assert not complete_checkpoint(path,50)
    torch.save({'epoch':49,'epoch_complete':True},path)
    assert complete_checkpoint(path,50)
    base={'seed':0,'epochs':50,'data_root':'/content/data','output':'/content/run','workers':2}
    assert signature(base)==signature({**base,'data_root':'/tmp/data','output':'/kaggle/run','workers':1})
    assert signature(base)!=signature({**base,'epochs':1})


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA is required for FP16 integration')
@pytest.mark.parametrize('kind',['edge','gcn','sage','gatv2'])
def test_rgb_ctc_loss_backward_with_real_cuda_amp(kind):
    cfg={**BASE,'graph_conv':kind,'hidden_size':16,'allow_random_init':True,'distillation':1.}
    model=SignModel(cfg,['<blank>','<unk>','a']).cuda()
    video=torch.randn(1,16,3,224,224,device='cuda')
    with torch.autocast('cuda',dtype=torch.float16):
        loss=model(video,torch.tensor([16]),[{'id':'synthetic','gloss':'a'}])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_incomplete_session_cannot_evaluate_or_mark_completed(tmp_path,monkeypatch):
    import scripts.run_suite as suite
    from scripts.run_local import ROOT
    manifest=tmp_path/'manifests'; manifest.mkdir()
    for split in ('train','dev','test'): (manifest/f'{split}.jsonl').write_text('{"id":"sample"}\n')
    args=SimpleNamespace(only='main',output=tmp_path/'out',manifests=manifest,data_root=tmp_path/'data',
        seed=0,epochs=50,workers=0,minutes=0,checkpoint_every=1)
    class Remote:
        root=None
        def folder(self,*args): return None
        def get(self,*args): pass
        def put(self,*args): pass
        def store(self,*args): return None,None
    calls=[]
    monkeypatch.setattr(suite.subprocess,'run',lambda command,**kwargs: calls.append(command))
    class Process:
        returncode=0
        def __init__(self,command,**kwargs):
            calls.append(command)
            torch.save({'epoch':0,'epoch_complete':False},args.output/'main/last.pt')
        def poll(self): return 0
    monkeypatch.setattr(suite.subprocess,'Popen',Process)
    execute(args,Remote())
    assert not (args.output/'main/COMPLETED.json').exists()
    assert not any('--mode' in c for c in calls)
    summary=json.loads((args.output/'suite_summary.json').read_text())
    assert summary[0]['status']=='pending'


def test_visualizer_exports_actual_edges_and_declares_fallback_sample(tmp_path,monkeypatch):
    import yaml
    from PIL import Image
    from scripts.visualize_suite import render
    data=tmp_path/'data'; frames=data/'frames'; frames.mkdir(parents=True)
    for index in range(3): Image.new('RGB',(64,64),(20+index,50,90)).save(frames/f'{index:04d}.png')
    manifest=tmp_path/'manifests'; manifest.mkdir()
    (manifest/'test.jsonl').write_text(json.dumps({'id':'sample','frames':'frames','gloss':'a'})+'\n')
    output=tmp_path/'campaign'; main=output/'main'; main.mkdir(parents=True)
    cfg={**BASE,'hidden_size':16,'allow_random_init':True,'data_root':str(data),'test':str(manifest/'test.jsonl')}
    model=SignModel(cfg,['<blank>','<unk>','a'])
    torch.save({'model':model.state_dict(),'vocab':['<blank>','<unk>','a']},main/'best.pt')
    (main/'suite-config.yaml').write_text(yaml.safe_dump(cfg))
    (main/'test_predictions.json').write_text(json.dumps([{'id':'sample','reference':'a','sequence':'a'}]))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    render(output)
    qualitative=json.loads((output/'qualitative.json').read_text())
    assert qualitative['selection']=='First test sample: paper sample not found'
    edges=json.loads((output/'graph_edges.json').read_text())['edges']
    for scale in (16,32):
        assert len(edges[str(scale)]['tsg'][0])==49
        with Image.open(output/f'graph_stage{scale}.png') as image:
            assert image.size==(448,744); image.verify()


def test_drive_campaign_path_is_shared_with_colab(tmp_path,monkeypatch):
    import scripts.kaggle_drive as drive
    from scripts.run_suite import Remote
    credentials=tmp_path/'private.json'; credentials.write_text('{}')
    monkeypatch.setenv('SIGN_DRIVE_CREDENTIALS',str(credentials))
    monkeypatch.setattr(drive,'service_from_json',lambda value:object())
    calls=[]
    def folder(service,parent,name):
        calls.append((parent,name)); return 'runs-id' if name=='runs' else 'campaign-id'
    monkeypatch.setattr(drive,'ensure_folder',folder)
    remote=Remote(SimpleNamespace(drive_folder='shared-id',output=tmp_path/'phoenix14t_cslr_suite_seed0',persist_root=None))
    assert calls==[('shared-id','runs'),('runs-id','phoenix14t_cslr_suite_seed0')]
    assert remote.root=='campaign-id'


def test_temporal_augmentation_preserves_ctc_alignment(tmp_path,monkeypatch):
    import random
    from PIL import Image
    from repro.data import SignDataset
    frames=tmp_path/'frames'; frames.mkdir()
    for i in range(14): Image.new('RGB',(32,32)).save(frames/f'{i:04d}.png')
    manifest=tmp_path/'train.jsonl'
    manifest.write_text(json.dumps({'id':'short','frames':'frames','gloss':'a a b'})+'\n')
    monkeypatch.setattr(random.Random,'uniform',lambda *args:.8)
    video,_=SignDataset(manifest,tmp_path,train=True)[0]
    # a a b requires 4 CTC timesteps, so at least 13 original frames.
    assert len(video)==13
