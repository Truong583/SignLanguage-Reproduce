"""Real graph/backbone/CTC backward check, without dataset or downloads."""
import argparse
import json
import torch
from .data import collate
from .model import SignModel
from .runtime import environment,seed_everything

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--rgb',action='store_true',help='Also exercise full 224px backbone; uses more memory')
    a=p.parse_args()
    torch.set_num_threads(2)
    seed_everything(0)
    device=torch.device(a.device)
    cfg={'task':'cslr','input_kind':'rgb' if a.rgb else 'features','feature_dim':16,
         'hidden_size':32,'allow_random_init':True,'hsg':True,'distillation':1.}
    vocab=['<blank>','<unk>','a','b']
    model=SignModel(cfg,vocab).to(device)
    batch=collate([(torch.rand((8,3,224,224) if a.rgb else (8,16)),{'id':'synthetic','gloss':'a b'})])
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        loss=model(batch['video'].to(device),batch['lengths'],batch['rows'])
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Nonfinite smoke gradients')
    optimizer.step()
    if a.rgb:
        for name in ['hsg1','hsg2']:
            if not any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model.backbone,name).parameters()):
                raise RuntimeError(f'Missing {name} gradient')
    print(json.dumps({'smoke':'PASS','rgb':a.rgb,'loss':loss.item(),'environment':environment(),
                      'paper_result':False,'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None},indent=2))

if __name__=='__main__': main()
