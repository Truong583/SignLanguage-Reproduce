import gc
import json
import os
from types import SimpleNamespace

import pytest
import torch

from repro.model import SignModel
from scripts import report_wandb


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required to verify GPU/CPU activation transfers')
@pytest.mark.parametrize('precision',['fp32','fp16'])
@pytest.mark.parametrize('storage',['cpu','cpu_checkpoint'])
def test_real_rgb_ctc_offload_matches_loss_gradients_bn_rng_and_reduces_gpu_memory(precision,storage,monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(51)
    torch.backends.cudnn.benchmark=False
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(torch.backends.cudnn,'deterministic',True)
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',False)
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',False)
    cfg={'task':'cslr','input_kind':'rgb','hidden_size':16,'allow_random_init':True,
         'hsg':True,'graph_conv':'edge','distillation':1.,'activation_offload':'none'}
    model=SignModel(cfg,['<blank>','<unk>','a']).cuda().train()
    # cuDNN keeps an internal RNN dropout state which manual_seed does not
    # rewind between forwards. Remove that confound only in this paired test;
    # production training keeps the configured dropout of 0.3.
    model.temporal.rnn.dropout=0.
    initial={name:tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()}
    video=torch.randn(1,20,3,224,224,device='cuda')
    lengths=torch.tensor([20]); rows=[{'id':'sample','gloss':'a'}]
    results=[]
    for mode in ('none','none',storage):
        model.load_state_dict(initial); model.zero_grad(set_to_none=True)
        model.activation_offload=mode
        torch.manual_seed(91)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.float16,enabled=precision=='fp16'):
            loss=model(video,lengths,rows)
        assert torch.isfinite(loss)
        loss.backward(); torch.cuda.synchronize()
        grads={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
        buffers={name:value.detach().cpu().clone() for name,value in model.named_buffers()}
        results.append((loss.item(),grads,buffers,torch.cuda.get_rng_state(),torch.cuda.max_memory_allocated()))
        del loss
    reference,repeated,actual=results
    assert actual[0]==pytest.approx(reference[0],rel=2e-4,abs=2e-4)
    assert actual[1].keys()==reference[1].keys()
    for name in reference[1]:
        if precision=='fp32':
            torch.testing.assert_close(actual[1][name],reference[1][name],rtol=3e-3,atol=2e-4)
        else:
            # max_pool3d backward has no deterministic CUDA implementation.
            # Bound per-parameter error by 1% norm plus measured repeat noise,
            # instead of asserting bitwise equality of half-precision sums.
            delta=(actual[1][name]-reference[1][name]).float().norm()
            noise=(repeated[1][name]-reference[1][name]).float().norm()
            scale=reference[1][name].float().norm()
            assert delta<=.001+.01*scale+2*noise,name
        assert torch.isfinite(actual[1][name]).all()
    for name in reference[2]: torch.testing.assert_close(actual[2][name],reference[2][name],rtol=0,atol=0)
    assert torch.equal(actual[3],reference[3])
    assert actual[4]<reference[4]*.9
    print(json.dumps({'precision':precision,'baseline_peak_mib':reference[4]/1024**2,
                      'activation_storage':storage,'storage_peak_mib':actual[4]/1024**2}))
    del model,video,results
    gc.collect(); torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_edgeconv_offload_preserves_fp16_output_input_and_parameter_gradients():
    from repro.graphs import EdgeConv,hierarchical_edges
    torch.manual_seed(7)
    layer=EdgeConv(16).cuda(); edges=hierarchical_edges(4,4,2,2,'cuda')
    values=torch.randn(3,20,16,device='cuda'); records=[]
    for offload in (False,True):
        from contextlib import nullcontext
        x=values.detach().clone().requires_grad_(); layer.zero_grad(set_to_none=True)
        with torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else nullcontext():
            with torch.autocast('cuda',dtype=torch.float16): output=layer(x,edges)
        output.float().square().mean().backward()
        records.append((output.detach().clone(),x.grad.clone(),[p.grad.clone() for p in layer.parameters()]))
    torch.testing.assert_close(records[0][0],records[1][0],rtol=0,atol=0)
    torch.testing.assert_close(records[0][1],records[1][1],rtol=1e-3,atol=1e-5)
    for a,b in zip(records[0][2],records[1][2]): torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_long_rgb_main_hidden_size_fp16_has_finite_gradients_and_adam_update():
    torch.set_num_threads(2); torch.manual_seed(9)
    cfg={'task':'cslr','input_kind':'rgb','hidden_size':1024,'allow_random_init':True,
         'hsg':True,'graph_conv':'edge','distillation':25.}
    vocab=['<blank>','<unk>','a']+[f'gloss{i}' for i in range(1293)]
    model=SignModel(cfg,vocab).cuda().train()
    assert model.activation_offload=='cpu_checkpoint' and model.temporal.rnn.dropout==.3
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    frames=int(os.environ.get('SIGNLANGUAGE_STRESS_FRAMES','128'))
    video=torch.randn(1,frames,3,224,224,device='cuda')
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.float16):
        loss=model(video,torch.tensor([frames]),[{'id':'long','gloss':'a'}])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(int(value)==1 for name,value in model.named_buffers() if name.endswith('num_batches_tracked'))
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    before=model.classifier.weight.detach().clone()
    optimizer.step(); torch.cuda.synchronize()
    assert torch.isfinite(model.classifier.weight).all() and not torch.equal(before,model.classifier.weight)
    import psutil
    print(json.dumps({'main_hidden_size':1024,'frames':frames,'peak_vram_mib':torch.cuda.max_memory_allocated()/1024**2,
                      'rss_mib':psutil.Process().memory_info().rss/1024**2,
                      'peak_rss_mib':getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)/1024**2,
                      'vocab_size':len(vocab),'loss':loss.item()}))
    del loss,optimizer,model,video
    gc.collect(); torch.cuda.empty_cache()


def test_cpu_training_does_not_enter_cuda_offload_context(monkeypatch):
    def unexpected(*args,**kwargs): raise AssertionError('CPU execution must not offload')
    monkeypatch.setattr(torch.autograd.graph,'save_on_cpu',unexpected)
    model=SignModel({'input_kind':'features','feature_dim':8,'hidden_size':16},['<blank>','<unk>','a'])
    loss=model(torch.randn(1,16,8),torch.tensor([16]),[{'id':'cpu','gloss':'a'}])
    assert torch.isfinite(loss); loss.backward()


def test_offload_mode_is_validated():
    with pytest.raises(ValueError,match='activation_offload'):
        SignModel({'activation_offload':'invalid'},['<blank>','<unk>','a'])


def test_observer_finishes_sdk_session_when_stopped_between_uploads(tmp_path,monkeypatch):
    root=tmp_path/'runs'; root.mkdir(); spool=tmp_path/'spool'; spool.mkdir()
    finished=[]
    run=SimpleNamespace(url='https://example.test/private-run',summary={},finish=lambda **kw:finished.append(kw))
    sdk=SimpleNamespace(Settings=lambda **kw:kw,init=lambda **kw:run)
    monkeypatch.setattr(report_wandb,'require_private',lambda *args:None)
    def stop(_): raise KeyboardInterrupt
    monkeypatch.setattr(report_wandb.time,'sleep',stop)
    report_wandb.observe(root,spool,{'wandb_entity':'entity','wandb_project':'project'},'dummy',sdk)
    assert finished==[{'exit_code':0}]
    assert json.loads((spool/'health.json').read_text())['status']=='stopped'
